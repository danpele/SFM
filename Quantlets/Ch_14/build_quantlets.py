"""
build_quantlets.py -- Quantlet folders of Chapter 14 (SFM): crypto assets and stablecoins
=========================================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/sfm_quantlets.py).
The notebooks install the arch package first (pip install arch), as Google Colab does not include it.
Run:  python3 Quantlets/Ch_14/generate_all_charts.py && python3 Quantlets/Ch_14/seminar14.py
      python3 Quantlets/Ch_14/build_quantlets.py
      python3 notebooks/split_seminar_notebooks.py 14
Statistics of Financial Markets - Daniel Traian PELE
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
sys.path.insert(0, HERE)
import generate_all_charts as g                 # noqa: E402
from sfm_quantlets import build_all             # noqa: E402

SUBMITTED = 'Saturday, 3 October 2026'
INSTALL = '# the arch package (maximum-likelihood estimation of GARCH models) is not part of Google Colab\n!pip install -q arch'
DATA = ('Daily market data from EODHD (Bitcoin, Ethereum, Solana, USDT, USDC, DAI, S&P 500, gold XAU/USD, the iShares '
        'Bitcoin Trust ETF IBIT), data/market of the SFM repository; supply and prices of USD stablecoins from the '
        'public DefiLlama API (stablecoins.llama.fi)')
CONSTS = ['import matplotlib.dates as mdates', 'import statsmodels.api as sm', 'from arch import arch_model',
          f'EXTRA = {g.EXTRA!r}', f'NAME = {g.NAME!r}', f'START = {g.START!r}', f'ASSETS = {g.ASSETS!r}',
          f'COLORS = {g.COLORS!r}', f'HALVINGS = {g.HALVINGS!r}', f'ETF_DATE = {g.ETF_DATE!r}', f'EVENTS = {g.EVENTS!r}',
          f'EPISODES = {g.EPISODES!r}', f'HILL_FRAC = {g.HILL_FRAC!r}', f'VR_Q = {g.VR_Q!r}',
          f'VR_WINDOW, VR_STEP = {g.VR_WINDOW!r}, {g.VR_STEP!r}', f'SUBPERIODS = {g.SUBPERIODS!r}',
          f'CORR_WINDOW = {g.CORR_WINDOW!r}', f'ALPHA_VAR, ALPHA_ES = {g.ALPHA_VAR!r}, {g.ALPHA_ES!r}',
          f'POSITION = {g.POSITION!r}', f'BT_START = {g.BT_START!r}', f'BT_WINDOW = {g.BT_WINDOW!r}',
          f'PEG_START = {g.PEG_START!r}', f'SVB = {g.SVB!r}', f'LLAMA = {g.LLAMA!r}', f'STABLE_ID = {g.STABLE_ID!r}',
          f'DAYS = {g.DAYS!r}', f'WD_PERIODS = {g.WD_PERIODS!r}', f'SEED = {g.SEED!r}', '_LLAMA = {}', "TABLE_DIR = '.'"]
BASE = [g.register_extra, g.close, g.returns, g.joint_returns]

QUANTLETS = [
    dict(name='SFM_ch14_bitcoin_basics',
         desc='The Bitcoin supply rule: a block subsidy of 50 BTC halved every 210 000 blocks (about four years) gives a '
              'supply limit of 21 million coins; the supply path from the protocol rule; the Bitcoin price since 2014 (log '
              'scale) with the halvings of 2016, 2020 and 2024 and the price one year after each halving.',
         keywords='Bitcoin, blockchain, proof of work, halving, block subsidy, supply schedule, geometric series',
         consts=CONSTS, funcs=BASE + [g.btc_supply_schedule, g.fig_halving],
         run="print(btc_supply_schedule().round(3))\nprint(fig_halving())",
         charts=['sfm_ch14_halving']),
    dict(name='SFM_ch14_asset_class',
         desc='Crypto as an asset class: descriptive statistics of daily log returns of Bitcoin, Ethereum, Solana, the S&P '
              '500 and gold; annualisation with the actual frequency (365 days a year for crypto) against 252 days; Hill '
              'tail indices of the left and right tails (k = 2.5% of n) with 95% intervals; volatility on monthly rolling '
              'windows; the weekday effect in the mean (weekend dummy, HAC standard errors) and in the variance '
              '(Brown-Forsythe test) in three periods.',
         keywords='cryptocurrency, Bitcoin, Ethereum, annualisation, volatility, heavy tails, Hill estimator, weekend effect, '
                  'day-of-the-week effect, Brown-Forsythe test, HAC standard errors',
         consts=CONSTS, funcs=BASE + [g.hill, g.describe, g.stats_table, g.fig_rolling_vol, g.hac_ols, g.weekday_effect,
                                      g.fig_weekday],
         run="T = stats_table()\nprint(pd.DataFrame(T).T[['n', 'mean', 'sd', 'skew', 'exkurt', 'ann_vol', 'ann_vol_252', "
             "'hill_left', 'hill_right']].round(3))\nprint(fig_rolling_vol())\n"
             "W = fig_weekday()\nprint({k: (round(v['b'], 3), round(v['p'], 3), round(v['var_ratio'], 3), v['p_bf']) for k, v in W.items()})",
         charts=['sfm_ch14_rolling_vol', 'sfm_ch14_weekday'], extra=['ch14_stats_table.csv']),
    dict(name='SFM_ch14_garch_efficiency',
         desc='Volatility clustering and efficiency of crypto markets: ACF of Bitcoin returns and squared returns, Ljung-Box '
              'tests; GARCH(1,1) and GJR-GARCH(1,1) with Student-t innovations for Bitcoin, Ethereum, the S&P 500 and gold '
              '(persistence, half-life, asymmetry); conditional volatility; Lo-MacKinlay variance ratios VR(2), VR(5), '
              'VR(10) with heteroskedasticity-robust Z* statistics on the whole sample, in sub-periods and on rolling '
              'windows of 500 observations, against the i.i.d. Z statistic.',
         keywords='volatility clustering, ACF, Ljung-Box, GARCH, IGARCH, GJR-GARCH, leverage effect, variance ratio, random '
                  'walk, weak-form efficiency, Bitcoin, Ethereum',
         consts=CONSTS, funcs=BASE + [g.acf, g.ljung_box, g.fig_acf, g.garch_fit, g.garch_table, g.fig_garch_vol,
                                      g.variance_ratio, g.efficiency_table, g.rolling_vr, g.fig_rolling_vr],
         run="print(fig_acf())\nG = garch_table()\nprint(pd.DataFrame(G).T[['alpha', 'beta', 'pers', 'half_life', 'nu', "
             "'gamma', 't_gamma']].round(3))\nfig_garch_vol()\nE = efficiency_table()\n"
             "print({k: {q: round(v['zs'], 2) for q, v in e['vr'].items()} for k, e in E.items()})\nprint(fig_rolling_vr())",
         charts=['sfm_ch14_acf', 'sfm_ch14_garch_vol', 'sfm_ch14_rolling_vr']),
    dict(name='SFM_ch14_correlation',
         desc='Correlation of crypto with equities and gold: prices joined on common days, daily and weekly log returns, '
              'rolling 250-day correlation of Bitcoin with the S&P 500 and gold and of Ethereum with the S&P 500, '
              'correlations by year, before and after 2020; cumulative returns in the COVID-19 (2020), Terra/Luna and FTX '
              '(2022) episodes; mean returns of Bitcoin and gold on the worst 5% of S&P 500 days (hedge and safe haven, '
              'Baur and Lucey, 2010).',
         keywords='correlation, rolling correlation, asynchronous trading, weekly returns, crisis, COVID-19, Terra, FTX, '
                  'safe haven, hedge, Bitcoin, gold, S&P 500',
         consts=CONSTS, funcs=BASE + [g.fig_rolling_corr, g.crisis_table, g.worst_days],
         run="C = fig_rolling_corr()\nprint({k: (round(v['pre2020'], 3), round(v['post2020'], 3), round(v['weekly'], 3)) for k, v in C.items()})\n"
             "print(pd.DataFrame(crisis_table()).T)\nprint(worst_days())",
         charts=['sfm_ch14_rolling_corr']),
    dict(name='SFM_ch14_bubbles_etf',
         desc='Bubbles, crashes and the spot Bitcoin ETFs: drawdowns of Bitcoin, Ethereum and the S&P 500, the episodes in '
              'which Bitcoin lost more than half of its value with their recovery dates; Bitcoin two years before and two '
              'years after the US spot ETFs of 11 January 2024 (annualised volatility, Brown-Forsythe test, weekend share '
              'of variance, correlation with the S&P 500, GARCH persistence); IBIT against Bitcoin (correlation, slope, '
              'tracking error).',
         keywords='drawdown, maximum drawdown, bubble, crash, spot Bitcoin ETF, IBIT, tracking error, event study, '
                  'Brown-Forsythe test, weekend effect',
         consts=CONSTS, funcs=BASE + [g.garch_fit, g.drawdown, g.drawdown_episodes, g.fig_drawdowns, g.etf_effect, g.fig_etf],
         run="D = fig_drawdowns()\nprint(pd.DataFrame(D['btc_episodes']))\nprint(etf_effect())\nprint(fig_etf())",
         charts=['sfm_ch14_drawdowns', 'sfm_ch14_etf']),
    dict(name='SFM_ch14_stablecoins',
         desc='Stablecoins: supply of USD stablecoins, USDT, USDC and DAI (DefiLlama); peg deviations in basis points, '
              'd = 10 000 (P - 1), on daily closes and lows since 2021, with the AR(1) half-life of a deviation; the USDC '
              'and DAI depeg of March 2023 after the closure of Silicon Valley Bank; the collapse of TerraUSD (UST) in May '
              '2022; weekly changes in stablecoin supply against Bitcoin returns (HAC regression).',
         keywords='stablecoin, peg, depeg, basis points, USDT, USDC, DAI, TerraUSD, algorithmic stablecoin, Silicon Valley '
                  'Bank, DefiLlama, half-life, AR(1)',
         consts=CONSTS, funcs=BASE + [g.hac_ols, g.stablecoin_chart, g.stablecoin_prices, g.fig_stablecoin_supply,
                                      g.peg_stats, g.fig_peg, g.fig_svb, g.fig_ust, g.stablecoin_flows_btc],
         run="print(fig_stablecoin_supply())\nprint(pd.DataFrame(fig_peg()).T.round(3))\nprint(fig_svb())\nprint(fig_ust())\n"
             "print(stablecoin_flows_btc())",
         charts=['sfm_ch14_stablecoin_supply', 'sfm_ch14_peg', 'sfm_ch14_svb', 'sfm_ch14_ust']),
    dict(name='SFM_ch14_crypto_var',
         desc='VaR 1% and ES 2.5% (VaR_alpha = -q_alpha, the loss exceeded with probability alpha) of a position of 100 000 '
              'USD in Bitcoin, Ethereum and the S&P 500 by historical simulation, Normal, Student-t (maximum likelihood) and '
              'GARCH(1,1)-t for the next day; conversion of log-return VaR into USD; 10-day VaR against the '
              'square-root-of-time rule; a backtest of one-day Bitcoin VaR 1% since 2018 (historical simulation on 365 days '
              'and GARCH(1,1)-t re-estimated every 365 days) with the Kupiec test.',
         keywords='value at risk, expected shortfall, crypto, Bitcoin, historical simulation, Student-t, GARCH, backtesting, '
                  'Kupiec test',
         consts=CONSTS, funcs=BASE + [g.garch_fit, g.hs_var, g.hs_es, g.std_t_q, g.std_t_es, g.var_table, g.kupiec,
                                      g.backtest_btc, g.fig_btc_var],
         run="V = var_table()\nprint(pd.DataFrame(V).T[['hs_v', 'hs_e', 'n_v', 't_v', 'g_v', 'hs_v_usd', 'hs10', 'sqrt10']].round(2))\n"
             "bt, df = backtest_btc()\nprint(bt)\nfig_btc_var(df)",
         charts=['sfm_ch14_btc_var']),
]

try:                                   # seminar code (instructor files, see .gitignore)
    import seminar14 as s
    QUANTLETS.append(dict(
        name='SFM_ch14_seminar',
        desc='Seminar 14 of Statistics of Financial Markets: annualising crypto returns with 365 and 252 days, the drawdown '
             'of a price path, peg deviations of USDC and DAI in March 2023 in basis points, VaR 1% and ES 2.5% of a crypto '
             'position, on paper; Bitcoin against the S&P 500 (moments, annualisation, Hill tail index), the correlation '
             'with equities before and after 2020 (Fisher z test) and in crises, the peg of USDC (AR(1) half-life), VaR '
             'and a backtest of a Bitcoin position.',
        keywords='seminar, cryptocurrency, stablecoin, annualisation, drawdown, peg deviation, Hill estimator, correlation, '
                 'Fisher z test, VaR, expected shortfall, backtesting',
        consts=CONSTS, funcs=BASE + [g.hill, g.describe, g.hac_ols, g.peg_stats, g.hs_var, g.hs_es, g.std_t_q, g.std_t_es,
                                     g.garch_fit, g.kupiec, g.backtest_btc, g.crisis_table, g.worst_days,
                                     s.annualise, s.drawdown_path, s.peg_bp, s.normal_crypto, s.worst_returns, s.hill_plot,
                                     s.b1_tails, s.fisher_diff, s.b3_corr, s.b5_peg, s.var_position, s.b7_var],
        consts_extra=None,
        run="print(annualise(0.12, 3.5))\nprint(drawdown_path([20, 35, 69, 50, 30, 16, 25, 45, 73]))\nprint(peg_bp('usdc'))\n"
            "print(normal_crypto(100_000, 0.10, 3.5))\nprint(worst_returns('btc'))\nprint(b1_tails())\nprint(b3_corr())\n"
            "print(b5_peg())\nprint(b7_var())",
        charts=['ch14_sem_b1', 'ch14_sem_b3', 'ch14_sem_b5', 'ch14_sem_b7']))
    QUANTLETS[-1]['consts'] = CONSTS + [f'BINS = {s.BINS!r}', f'BIN_LAB = {s.BIN_LAB!r}']
    QUANTLETS[-1].pop('consts_extra')
except ImportError:
    pass

if __name__ == '__main__':
    build_all(QUANTLETS, 14, 'Crypto assets', HERE, data=DATA, submitted=SUBMITTED, install=INSTALL)
