"""Export the inputs of the Chapter 14 interactive page (SFM/interactive/ch14_data.js).

Calls the functions of generate_all_charts.py (same data files in data/market, same start dates, same end date
2026-09-18), so the page starts from the numbers of the slides. Exported: the daily closes of Bitcoin, Ethereum, Solana,
the S&P 500, gold and the IBIT ETF (the browser recomputes from them the returns, the descriptive statistics, the Hill
indices, the rolling volatility, the weekday tests, the ACF and Ljung-Box tests, the variance ratios, the rolling and yearly
correlations, the crisis returns, the worst-day test, the drawdowns and their episodes, the ETF comparison, the tracking
error, the historical and parametric VaR and ES and the backtest of the historical VaR); the GARCH(1,1)-t and GJR estimates
and the conditional volatilities of Bitcoin and the S&P 500 (not refitted in the browser); the Student-t fits and the
GARCH-t one-day forecasts of the VaR table; the GARCH-t VaR 1% forecasts of the rolling backtest, with the sigma, mu and nu of
each refit; the daily closes and lows of USDT, USDC and DAI since 2021; the DefiLlama supply of USD stablecoins and the
supply and price of TerraUSD (downloaded with the chapter's loader, cut at 2026-09-18).
Extra runs of the chapter's code (labelled on the page): GARCH(1,1)-t persistence before and after the spot ETFs on
windows of 1, 1.5 and 2.5 years.
The script checks every number of ch14_numbers.json computed from the local data against the chapter functions (to 1e-12
relative, exact for counts, dates and labels) and stops if one no longer matches. The DefiLlama series are downloaded again:
their differences from ch14_numbers.json are written to the data file (the page shows the download value next to the slide's)
and printed, not corrected. It writes only interactive/ch14_data.js (no chart, no table of the chapter).
Run:  python3 Quantlets/Ch_14/export_interactive.py      (about one minute)
"""
import datetime
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import generate_all_charts as g
import sfm_style as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
OUT = os.path.join(ROOT, 'interactive', 'ch14_data.js')


def tree(a, b, what, tol=1e-12, soft=None):
    """a (recomputed) against b (ch14_numbers.json): floats to tol (relative), the rest exactly. With soft (a list), a
    mismatch is recorded there instead of stopping."""
    def bad(msg):
        if soft is None:
            raise SystemExit(msg)
        soft.append(msg)
    if isinstance(b, dict):
        A = {str(k): v for k, v in a.items()}
        if set(A) != set(b):
            return bad(f'{what}: keys {sorted(A)} do not match ch14_numbers.json {sorted(b)}')
        for k in b:
            tree(A[k], b[k], f'{what}/{k}', tol, soft)
    elif isinstance(b, list):
        if len(a) != len(b):
            return bad(f'{what}: length {len(a)} does not match ch14_numbers.json ({len(b)})')
        for i, (x, y) in enumerate(zip(a, b)):
            tree(x, y, f'{what}[{i}]', tol, soft)
    elif b is None or isinstance(b, (bool, str)) or (isinstance(b, int) and not isinstance(a, (float, np.floating))):
        if a != b and not (isinstance(a, (int, np.integer)) and int(a) == b):
            bad(f'{what}: {a} does not match ch14_numbers.json ({b})')
    else:
        a, b = float(a), float(b)
        if math.isnan(b) and math.isnan(a):
            return
        if not abs(a - b) <= tol * max(1.0, abs(b)):
            bad(f'{what}: {a} does not match ch14_numbers.json ({b})')


def rd(a, d=10):
    return [float(f'{x:.{d}g}') for x in np.asarray(a, float)]


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.floating):
        x = float(x)
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def ser(s, d=None):
    """A dated series as {t0, t (day offsets), v}."""
    s = s.dropna()
    t0 = s.index[0]
    return dict(t0=t0.date().isoformat(), t=[int(v) for v in (s.index - t0).days],
                v=[float(x) for x in s.values] if d is None else rd(s.values, d))


def sec_local(N):
    """Every number of ch14_numbers.json that comes from data/market, recomputed by the chapter functions."""
    tree(g.fig_halving(save=False), N['halving'], 'halving')
    tree({k: g.describe(k) for k in g.ASSETS}, N['stats'], 'stats')
    tree(g.fig_rolling_vol(save=False), N['rvol'], 'rvol')
    tree(g.weekday_effect(), N['weekday'], 'weekday')
    tree(g.fig_acf(save=False), N['acf'], 'acf')
    G = g.garch_table()
    tree(G, N['garch'], 'garch')
    tree(g.efficiency_table(), N['eff'], 'eff')
    tree(g.fig_rolling_vr(save=False), N['rvr'], 'rvr')
    tree(g.fig_rolling_corr(save=False), N['corr'], 'corr')
    tree(g.crisis_table(), N['crisis'], 'crisis')
    tree(g.worst_days(), N['worst'], 'worst')
    tree(g.fig_drawdowns(save=False), N['dd'], 'dd')
    tree(g.etf_effect(), N['etf'], 'etf')
    tree(g.fig_etf(save=False), N['etf_fig'], 'etf_fig')
    tree(g.fig_peg(save=False), N['peg'], 'peg')
    tree(g.fig_svb(save=False), N['svb'], 'svb')
    tree(g.var_table(), N['var'], 'var')
    bt, df = g.backtest_btc()
    tree(bt, N['bt'], 'bt')
    tree(g.returns('btc').index[-1].date().isoformat(), N['end'], 'end')
    plt.close('all')
    return G, df


def sec_prices():
    P = {k: ser(g.close(k)) for k in ('btc', 'eth', 'sol', 'sp500', 'gold')}
    P['ibit'] = ser(g.close('ibit', g.ETF_DATE))
    for k, s in P.items():                                  # returns of the chapter = 100 diff(log close) of these closes
        r = (g.returns(k) if k != 'ibit' else g.returns('ibit', g.ETF_DATE)).values
        x = 100 * np.diff(np.log(np.array(s['v'])))
        if not np.array_equal(x, r):
            raise SystemExit(f'prices {k}: the exported closes do not give the returns of the chapter')
    return P


def sec_garch(G):
    out = {}
    for k in ('btc', 'eth', 'sp500', 'gold'):
        r = g.returns(k)
        res = g.garch_fit(r)
        p = res.params
        f = res.forecast(horizon=1, reindex=False).variance.values[-1, 0]
        sig = res.conditional_volatility
        if abs(sig.max() * np.sqrt(g.periods_per_year(r)) - G[k]['sig_max']) > 1e-9:
            raise SystemExit(f'garch {k}: the conditional volatility does not reproduce')
        e = float(r.values[-1] - p['mu'])
        f1 = p['omega'] + p['alpha[1]'] * e ** 2 + p['beta[1]'] * sig.values[-1] ** 2   # sigma^2_{T+1}
        if abs(f1 - f) > 1e-9 * f:
            raise SystemExit(f'garch {k}: the one-day forecast does not follow from the last fitted variance')
        out[k] = dict(G[k], s1=float(np.sqrt(f)))
        if k in ('btc', 'sp500'):
            out[k]['sig'] = ser(sig, 7)
    return out


def sec_etf(N):
    """GARCH(1,1)-t persistence of Bitcoin on windows of L years before and after 11 January 2024 (L = 2 in the lecture)."""
    t = pd.Timestamp(g.ETF_DATE)
    out = {}
    for L in (2, 1, 1.5, 2.5):
        m = int(round(12 * L))
        b0, b1 = (t - pd.DateOffset(months=m)).date().isoformat(), (t - pd.Timedelta(days=1)).date().isoformat()
        a0, a1 = g.ETF_DATE, (t + pd.DateOffset(months=m) - pd.Timedelta(days=1)).date().isoformat()
        pb = g.garch_fit(g.returns('btc', b0, b1)).params
        pa = g.garch_fit(g.returns('btc', a0, a1)).params
        out[f'{L:g}'] = dict(before=[b0, b1], after=[a0, a1], pers_before=float(pb['alpha[1]'] + pb['beta[1]']),
                             pers_after=float(pa['alpha[1]'] + pa['beta[1]']))
    tree({'pers_before': out['2']['pers_before'], 'pers_after': out['2']['pers_after']},
         {k: N['etf'][k] for k in ('pers_before', 'pers_after')}, 'etf/pers')
    assert out['2']['before'] == ['2022-01-11', '2024-01-10'] and out['2']['after'] == ['2024-01-11', '2026-01-10']
    return out


def sec_var(N):
    out = {}
    for k in ('btc', 'eth', 'sp500'):
        x = g.returns(k).values
        nu, loc, sc = stats.t.fit(x)
        res = g.garch_fit(g.returns(k))
        p = res.params
        s1 = float(np.sqrt(res.forecast(horizon=1, reindex=False).variance.values[-1, 0]))
        tree({'nu': nu, 'g_sigma': s1, 'g_nu': p['nu']}, {q: N['var'][k][q] for q in ('nu', 'g_sigma', 'g_nu')}, f'var/{k}')
        out[k] = dict(nu=float(nu), loc=float(loc), sc=float(sc), g_mu=float(p['mu']), g_sigma=s1, g_nu=float(p['nu']))
    return out


def sec_bt(df):
    """GARCH-t VaR 1% forecasts of backtest_btc with the sigma, mu and nu of each refit (re-estimated every 365 days)."""
    r = g.returns('btc')
    idx = np.where(r.index >= pd.Timestamp(g.BT_START))[0]
    sig, mu, nu, blk = np.empty(len(idx)), [], [], np.empty(len(idx), int)
    for b, j0 in enumerate(range(0, len(idx), g.BT_WINDOW)):
        i0 = idx[j0]
        am = g.arch_model(r, mean='Constant', vol='GARCH', p=1, q=1, dist='t')
        res = am.fit(disp='off', last_obs=r.index[i0], options={'maxiter': 2000})
        f = res.forecast(horizon=1, start=r.index[i0 - 1], reindex=False)
        s = np.sqrt(f.variance.values[:-1, 0])
        n = min(g.BT_WINDOW, len(idx) - j0)
        sig[j0:j0 + n] = s[:n]
        blk[j0:j0 + n] = b
        mu.append(float(res.params['mu']))
        nu.append(float(res.params['nu']))
    v = -(np.array(mu)[blk] + sig * np.array([g.std_t_q(q, g.ALPHA_VAR) for q in nu])[blk])
    if np.max(np.abs(v - df['GARCH-t'].values)) > 1e-10:
        raise SystemExit('backtest: sigma, mu and nu do not give the GARCH-t VaR of the chapter')
    return dict(t0=df.index[0].date().isoformat(), var=rd(df['GARCH-t'].values, 12), sig=rd(sig, 12), blk=blk.tolist(), mu=mu, nu=nu)


def sec_peg():
    g.register_extra()
    out = {}
    for k in ('usdt', 'usdc', 'dai'):
        d = g.read_market(g.SERIES[k][0]).loc[g.PEG_START:g.END]
        t0 = d.index[0]
        out[k] = dict(t0=t0.date().isoformat(), t=[int(v) for v in (d.index - t0).days], close=[float(x) for x in d['close']],
                      low=[float(x) for x in d['low']])
    return out


def sec_llama(N, drift):
    """DefiLlama supply (all USD stablecoins, USDT, USDC, DAI) and TerraUSD, downloaded now with the chapter's loader."""
    today = datetime.date.today().isoformat()
    S = g.fig_stablecoin_supply(save=False)
    tree(S, N['supply'], 'supply', soft=drift)
    U = g.fig_ust(save=False)
    tree(U, N['ust'], 'ust', soft=drift)
    plt.close('all')
    tot = g.stablecoin_chart()['value'].loc['2019-01-01':]
    parts = {k: g.stablecoin_chart(g.STABLE_ID[k])['value'].loc['2019-01-01':] for k in ('usdt', 'usdc', 'dai')}
    u = g.stablecoin_chart(g.STABLE_ID['ust']).loc['2022-04-01':'2022-06-15']
    u = u[u['supply'] > 0]['supply']
    px = g.stablecoin_prices('terrausd', '2022-04-01', '2022-06-15')
    return dict(download=today, supply=dict(total=ser(tot, 8), **{k: ser(s, 8) for k, s in parts.items()}), mech=S['mech_today'],
                supply_now=S, ust_now=U, ust=dict(supply=ser(u, 9), price=ser(px, 9)))


def main():
    N = json.load(open(os.path.join(HERE, 'ch14_numbers.json')))
    print('local numbers')
    G, df = sec_local(N)
    out = dict(end=g.END, colors=dict(g.COLORS, MainBlue=st.MainBlue, IDAred=st.IDAred, Forest=st.Forest, Amber=st.Amber,
                                      Orange=st.Orange, Purple=st.Purple, Teal=st.Teal, Crimson=st.Crimson, DarkText=st.DarkText),
               name=g.NAME, halvings=g.HALVINGS, etf_date=g.ETF_DATE, events=g.EVENTS, episodes=g.EPISODES, svb=g.SVB, N=N)
    print('prices')
    out['px'] = sec_prices()
    print('garch')
    out['garch'] = sec_garch(G)
    print('etf windows')
    out['etf'] = sec_etf(N)
    print('var')
    out['var'] = sec_var(N)
    print('backtest')
    out['bt'] = sec_bt(df)
    out['peg'] = sec_peg()
    print('DefiLlama')
    drift = []
    out['llama'] = sec_llama(N, drift)
    out['llama']['drift'] = drift
    for m in drift:
        print('  DefiLlama differs from the slides:', m)
    js = ('// Inputs of the Chapter 14 interactive page, written by Quantlets/Ch_14/export_interactive.py.\n'
          'window.CH14_DATA = ' + json.dumps(clean(out), separators=(',', ':'), allow_nan=False) + ';\n')
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write(js)
    print('wrote', os.path.normpath(OUT), f'{len(js) / 1e3:.0f} kB')


if __name__ == '__main__':
    main()
