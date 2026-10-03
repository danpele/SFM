r"""
ch14_common.py -- shared helpers of the Chapter 14 generators (lecture and seminar), SFM
========================================================================================
Numbers from Quantlets/Ch_14/ch14_numbers.json (generate_all_charts.py) and sem14_results.json (seminar14.py);
the clickable citations of Chapter 14 (DOIs checked against Crossref, official pages checked, 3 October 2026).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sfm_build import ROOT, Values   # noqa: E402
from ch1_common import T, date, put_date   # noqa: E402,F401

QL = os.path.join(ROOT, 'Quantlets', 'Ch_14')
QLURL = 'https://github.com/danpele/SFM/tree/main/Quantlets/Ch_14'
NAMES = {'btc': 'Bitcoin', 'eth': 'Ethereum', 'sol': 'Solana', 'sp500': 'S\\&P 500', 'gold': '⟦gold||aur⟧',
         'usdt': 'USDT', 'usdc': 'USDC', 'dai': 'DAI'}


def load():
    with open(os.path.join(QL, 'ch14_numbers.json')) as f:
        return json.load(f)


def load_sem():
    with open(os.path.join(QL, 'sem14_results.json')) as f:
        return json.load(f)


def pv(p, d=3):
    """A p-value in text mode: 3 decimals, or '< 0.001' (marked for the RO decimal comma)."""
    return '$<$\\,⁅0.001⁆' if p < 0.001 else '⁅' + f'{p:.{d}f}' + '⁆'


def money(x):
    return f'{x:,.0f}'.replace(',', '\\,')


def d(V, key, iso):
    """A date in both languages: @{key} inside T(en, ro) becomes the EN or the RO form."""
    put_date(V, key, iso)


def values(N):
    """All lecture numbers as @{key} values."""
    V = Values()
    d(V, 'end', N['end'])
    H = N['halving']
    for y, h in H['halvings'].items():
        V.raw(f'hv.{y}.p', money(h['price']))
        V.raw(f'hv.{y}.p1', money(h['price_1y']))
        V.put(f'hv.{y}.ch', h['change_1y'], 0)
    V.raw('hv.end', money(H['price_end']))
    V.raw('hv.start', money(H['price_start']))
    V.raw('hv.max', money(H['max_price']))
    d(V, 'hv.maxd', H['max_date'])
    V.put('hv.s24', H['supply_2024'], 1)
    V.put('hv.s30', H['supply_2030'], 1)
    for k, s in N['stats'].items():
        V.int(f's.{k}.n', s['n'])
        V.raw(f's.{k}.y0', s['first'][:4])
        V.put(f's.{k}.ppy', s['ppy'], 1)
        V.put(f's.{k}.mean', s['mean'], 3)
        V.put(f's.{k}.sd', s['sd'], 2)
        V.put(f's.{k}.skew', s['skew'], 2)
        V.put(f's.{k}.k', s['exkurt'], 1)
        V.put(f's.{k}.min', s['min'], 1)
        d(V, f's.{k}.mind', s['min_date'])
        V.put(f's.{k}.max', s['max'], 1)
        V.put(f's.{k}.am', s['ann_mean'], 1)
        V.put(f's.{k}.am252', s['ann_mean_252'], 1)
        V.put(f's.{k}.av', s['ann_vol'], 1)
        V.put(f's.{k}.av252', s['ann_vol_252'], 1)
        V.put(f's.{k}.sh5', s['share5'], 1)
        V.raw(f's.{k}.kh', str(s['k']))
        for c in ['hill_left', 'hill_right', 'hill_left_lo', 'hill_left_hi', 'hill_right_lo', 'hill_right_hi']:
            V.put(f's.{k}.{c}', s[c], 2)
    R = N['rvol']
    for k in ('btc', 'eth', 'sp500'):
        V.put(f'rv.{k}.med', R[k]['median'], 0)
        V.put(f'rv.{k}.max', R[k]['max'], 0)
        V.put(f'rv.{k}.last', R[k]['last'], 0)
        d(V, f'rv.{k}.maxd', R[k]['max_date'])
    V.put('rv.ratio', R['ratio_median'], 1)
    V.put('rv.below', R['share_btc_below_sp'], 1)
    for lab, w in N['weekday'].items():
        key = 'wd.' + lab[:4]
        V.int(f'{key}.n', w['n'])
        V.put(f'{key}.b', w['b'], 2)
        V.put(f'{key}.t', w['t'], 2)
        V.put(f'{key}.p', w['p'], 2)
        V.put(f'{key}.bf', w['bf'], 1)
        V.raw(f'{key}.pbf', pv(w['p_bf']))
        V.put(f'{key}.vr', w['var_ratio'], 2)
        V.put(f'{key}.sdwe', w['sd_we'], 2)
        V.put(f'{key}.sdwd', w['sd_wd'], 2)
        V.put(f'{key}.sat', w['sd']['Sat'], 2)
        V.put(f'{key}.thu', w['sd']['Thu'], 2)
    A = N['acf']
    V.put('acf.rho1', A['rho1'], 3)
    V.put('acf.rho1sq', A['rho1_sq'], 3)
    V.put('acf.rho10sq', A['rho10_sq'], 3)
    V.put('acf.band', A['band'], 3)
    V.raw('acf.nout', str(A['n_out_r']))
    V.raw('acf.noutsq', str(A['n_out_sq']))
    V.put('acf.lbr', A['lb_r']['q'], 1)
    V.raw('acf.lbrp', pv(A['lb_r']['p']))
    V.put('acf.lbsq', A['lb_sq']['q'], 1)
    V.raw('acf.lbsqp', pv(A['lb_sq']['p']))
    V.put('acf.splbr', A['sp_lb_r']['q'], 1)
    V.put('acf.splbsq', A['sp_lb_sq']['q'], 0)
    for k, g in N['garch'].items():
        for c in ['omega', 'alpha', 'beta', 'pers', 'gamma']:
            V.put(f'g.{k}.{c}', g[c], 3)
        V.put(f'g.{k}.nu', g['nu'], 2)
        V.put(f'g.{k}.tg', g['t_gamma'], 2)
        V.put(f'g.{k}.ta', g['t_alpha'], 1)
        V.raw(f'g.{k}.hl', '--' if g['half_life'] is None else '⁅' + f"{g['half_life']:.0f}" + '⁆')
        V.raw(f'g.{k}.unc', '--' if g['uncond_ann'] is None else '⁅' + f"{g['uncond_ann']:.1f}" + '⁆')
        V.put(f'g.{k}.last', g['sig_last'], 0)
        V.put(f'g.{k}.maxs', g['sig_max'], 0)
        d(V, f'g.{k}.maxd', g['sig_max_date'])
    E = N['eff']
    for k, e in E.items():
        V.put(f'e.{k}.rho1', e['rho1'], 3)
        V.put(f'e.{k}.lb', e['lb']['q'], 1)
        V.raw(f'e.{k}.lbp', pv(e['lb']['p']))
        for q, v in e['vr'].items():
            V.put(f'e.{k}.vr{q}', v['vr'], 3)
            V.put(f'e.{k}.zs{q}', v['zs'], 2)
            V.raw(f'e.{k}.p{q}', pv(v['p']))
        for lab, s in e['sub'].items():
            key = f'e.{k}.{lab[:4]}'
            V.int(f'{key}.n', s['n'])
            V.raw(f'{key}.lab', lab.replace('-', '--'))
            V.put(f'{key}.rho1', s['rho1'], 3)
            V.put(f'{key}.vr5', s['vr5']['vr'], 3)
            V.put(f'{key}.zs5', s['vr5']['zs'], 2)
            V.raw(f'{key}.p5', pv(s['vr5']['p']))
    for k, r in N['rvr'].items():
        V.raw(f'rvr.{k}.n', str(r['n_win']))
        V.put(f'rvr.{k}.rej', r['share_rej'], 1)
        V.put(f'rvr.{k}.iid', r['share_rej_iid'], 1)
        V.put(f'rvr.{k}.vmin', r['vr_min'], 2)
        V.put(f'rvr.{k}.vmax', r['vr_max'], 2)
    for pair, c in N['corr'].items():
        V.int(f'c.{pair}.n', c['n'])
        for f in ['all', 'weekly', 'pre2020', 'post2020', 'max', 'min', 'last']:
            V.put(f'c.{pair}.{f}', c[f], 2)
        d(V, f'c.{pair}.maxd', c['max_date'])
        for y, v in c['year'].items():
            V.put(f'c.{pair}.y{y}', v, 2)
    for lab, c in N['crisis'].items():
        key = 'cr.' + lab.split('/')[0].split('-')[0].lower()
        for k in ('btc', 'eth', 'sp500', 'gold'):
            V.put(f'{key}.{k}', c[k], 1)
        V.put(f'{key}.worst', c['btc_worst'], 1)
        d(V, f'{key}.wd', c['btc_worst_date'])
        d(V, f'{key}.a', c['start'])
        d(V, f'{key}.b', c['end'])
    for k, w in N['worst'].items():
        V.raw(f'w.{k}.nb', str(w['n_bad']))
        V.put(f'w.{k}.thr', w['thr'], 2)
        V.put(f'w.{k}.msp', w['mean_sp'], 2)
        V.put(f'w.{k}.m', w['mean'], 2)
        V.put(f'w.{k}.t', w['t'], 2)
        V.raw(f'w.{k}.p', pv(w['p']))
        V.put(f'w.{k}.neg', w['share_neg'], 0)
    D = N['dd']
    for k in ('btc', 'eth', 'sp500'):
        V.put(f'dd.{k}.mdd', D[k]['mdd'], 1)
        d(V, f'dd.{k}.mddd', D[k]['mdd_date'])
        V.put(f'dd.{k}.last', D[k]['last'], 1)
        V.put(f'dd.{k}.s10', D[k]['share_10'], 0)
    V.raw('dd.nep', str(len(D['btc_episodes'])))
    EF = N['etf']
    for c in ['vol_before', 'vol_after', 'we_share_before', 'we_share_after']:
        V.put(f'etf.{c}', EF[c], 1)
    for c in ['we_ratio_before', 'we_ratio_after', 'corr_before', 'corr_after', 'bf', 'pers_after']:
        V.put(f'etf.{c}', EF[c], 2)
    V.put('etf.p_bf', EF['p_bf'], 2)
    V.put('etf.pers_before', EF['pers_before'], 3)
    V.int('etf.nb', EF['n_before'])
    V.int('etf.na', EF['n_after'])
    I = EF['ibit']
    V.raw('ibit.n', str(I['n']))
    V.put('ibit.corr', I['corr'], 2)
    V.put('ibit.beta', I['beta'], 2)
    V.put('ibit.te', I['te'], 1)
    S = N['supply']
    for c in ['total_end', 'total_2020', 'total_2022max', 'total_trough', 'usdt_end', 'usdc_end', 'dai_end',
              'usdc_2023_03_10', 'usdc_2023_12_31', 'growth_x']:
        V.put(f'sup.{c}', S[c], 0 if S[c] > 20 else 1)
    V.put('sup.usdt_share', S['usdt_share'], 0)
    V.put('sup.usdc_share', S['usdc_share'], 0)
    d(V, 'sup.end', S['end'])
    d(V, 'sup.maxd', S['total_2022max_date'])
    d(V, 'sup.trd', S['total_trough_date'])
    M = S['mech_today']
    tot = sum(M.values())
    V.put('sup.fiat', 100 * M.get('fiat-backed', 0) / tot, 1)
    V.put('sup.crypto', 100 * M.get('crypto-backed', 0) / tot, 1)
    V.put('sup.algo', 100 * M.get('algorithmic', 0) / tot, 1)
    for k, p in N['peg'].items():
        V.put(f'peg.{k}.sd', p['sd'], 1)
        V.put(f'peg.{k}.mad', p['mad'], 1)
        V.put(f'peg.{k}.gt10', p['gt10'], 1)
        V.put(f'peg.{k}.gt50', p['gt50'], 2)
        V.put(f'peg.{k}.min', p['close_min'], 0)
        d(V, f'peg.{k}.mind', p['close_min_date'])
        V.put(f'peg.{k}.low', p['low_min'], 0)
        d(V, f'peg.{k}.lowd', p['low_min_date'])
        V.put(f'peg.{k}.phi', p['phi'], 2)
        V.put(f'peg.{k}.hl', p['half_life'], 1)
        V.int(f'peg.{k}.n', p['n'])
    for k, s in N['svb'].items():
        V.put(f'svb.{k}.low', s['low'], 4)
        V.put(f'svb.{k}.cmin', s['close_min'], 4)
        V.put(f'svb.{k}.cmax', s['close_max'], 4)
        V.put(f'svb.{k}.lowbp', 1e4 * (s['low'] - 1), 0)
        V.put(f'svb.{k}.cminbp', 1e4 * (s['close_min'] - 1), 0)
        V.put(f'svb.{k}.cmaxbp', 1e4 * (s['close_max'] - 1), 0)
    U = N['ust']
    V.put('ust.peak', U['supply_peak'], 1)
    d(V, 'ust.peakd', U['supply_peak_date'])
    V.put('ust.last', U['supply_last'], 1)
    d(V, 'ust.lastd', U['last_date'])
    V.put('ust.drop', U['supply_drop'], 0)
    for c in ['px_0509', 'px_0513', 'px_0514', 'px_0531']:
        V.put(f'ust.{c}', U[c], 2)
    V.put('ust.px_end', U['px_end'], 3)
    for k, v in N['var'].items():
        for c in ['hs_v', 'hs_e', 'n_v', 'n_e', 't_v', 't_e', 'g_v', 'g_e', 'hs10', 'sqrt10', 'g_sigma', 'sd']:
            V.put(f'v.{k}.{c}', v[c], 2)
        V.put(f'v.{k}.nu', v['nu'], 2)
        V.put(f'v.{k}.gnu', v['g_nu'], 2)
        for c in ['hs_v', 'hs_e', 'n_v', 'n_e', 't_v', 't_e', 'g_v', 'g_e', 'hs10']:
            V.raw(f'v.{k}.{c}.usd', money(v[c + '_usd']))
        V.put(f'v.{k}.gap', 100 * (1 - v['n_v'] / v['hs_v']), 0)
    B = N['bt']
    V.int('bt.n', B['n'])
    d(V, 'bt.start', B['start'])
    for m, key in (('HS', 'hs'), ('GARCH-t', 'g')):
        V.raw(f'bt.{key}.x', str(B[m]['x']))
        V.put(f'bt.{key}.rate', B[m]['rate'], 2)
        V.put(f'bt.{key}.exp', B[m]['exp'], 1)
        V.put(f'bt.{key}.lr', B[m]['lr'], 2)
        V.put(f'bt.{key}.p', B[m]['p'], 2)
        V.put(f'bt.{key}.mv', B[m]['mean_var'], 1)
        V.raw(f'bt.{key}.y2018', str(B[m]['year']['2018']))
        V.raw(f'bt.{key}.y2022', str(B[m]['year']['2022']))
    AI = N['ai']
    V.raw('ai.n', str(AI['n']))
    for c in ['corr_same', 'corr_next', 'b', 't', 'p']:
        V.put(f'ai.{c}', AI[c], 2)
    V.put('ai.r2', 100 * AI['r2'], 1)
    return V


# -----------------------------------------------------------------------------
# Clickable citations (DOIs verified via Crossref, official pages verified, 3 October 2026)
# -----------------------------------------------------------------------------
REFS = r"""
\newcommand{\refNakamoto}{\href{https://bitcoin.org/bitcoin.pdf}{Nakamoto (2008)}}
\newcommand{\refButerin}{\href{https://ethereum.org/en/whitepaper/}{Buterin (2014)}}
\newcommand{\refMerge}{\href{https://ethereum.org/en/roadmap/merge/}{ethereum.org: The Merge}}
\newcommand{\refLT}{\href{https://doi.org/10.1093/rfs/hhaa113}{Liu \& Tsyvinski (2021)}}
\newcommand{\refLTW}{\href{https://doi.org/10.1111/jofi.13119}{Liu, Tsyvinski \& Wu (2022)}}
\newcommand{\refGS}{\href{https://doi.org/10.1111/jofi.12903}{Griffin \& Shams (2020)}}
\newcommand{\refMS}{\href{https://doi.org/10.1016/j.jfineco.2019.07.001}{Makarov \& Schoar (2020)}}
\newcommand{\refGorton}{\href{https://doi.org/10.3386/w30796}{Gorton et al.\ (2022)}}
\newcommand{\refGZ}{\href{https://doi.org/10.2139/ssrn.3888752}{Gorton \& Zhang (2021)}}
\newcommand{\refLVN}{\href{https://doi.org/10.1016/j.jimonfin.2022.102777}{Lyons \& Viswanath-Natraj (2023)}}
\newcommand{\refLMS}{\href{https://doi.org/10.3386/w31160}{Liu, Makarov \& Schoar (2023)}}
\newcommand{\refMZZ}{\href{https://doi.org/10.3386/w33882}{Ma, Zeng \& Zhang (2025)}}
\newcommand{\refUrquhart}{\href{https://doi.org/10.1016/j.econlet.2016.09.019}{Urquhart (2016)}}
\newcommand{\refNC}{\href{https://doi.org/10.1016/j.econlet.2016.10.033}{Nadarajah \& Chu (2017)}}
\newcommand{\refTL}{\href{https://doi.org/10.1016/j.frl.2019.101382}{Tran \& Leirvik (2020)}}
\newcommand{\refCF}{\href{https://doi.org/10.1016/j.econlet.2015.02.029}{Cheah \& Fry (2015)}}
\newcommand{\refKatsiampa}{\href{https://doi.org/10.1016/j.econlet.2017.06.023}{Katsiampa (2017)}}
\newcommand{\refBL}{\href{https://doi.org/10.1111/j.1540-6288.2010.00244.x}{Baur \& Lucey (2010)}}
\newcommand{\refBHL}{\href{https://doi.org/10.1016/j.intfin.2017.12.004}{Baur, Hong \& Lee (2018)}}
\newcommand{\refCP}{\href{https://doi.org/10.1016/j.frl.2018.11.012}{Caporale \& Plastun (2019)}}
\newcommand{\refGHMO}{\href{https://doi.org/10.1016/j.jmoneco.2017.12.004}{Gandal et al.\ (2018)}}
\newcommand{\refHill}{\href{https://doi.org/10.1214/aos/1176343247}{Hill (1975)}}
\newcommand{\refBoll}{\href{https://doi.org/10.1016/0304-4076(86)90063-1}{Bollerslev (1986)}}
\newcommand{\refLM}{\href{https://doi.org/10.1093/rfs/1.1.41}{Lo \& MacKinlay (1988)}}
\newcommand{\refFama}{\href{https://doi.org/10.2307/2325486}{Fama (1970)}}
\newcommand{\refJLS}{\href{https://doi.org/10.1142/S0219024900000115}{Johansen, Ledoit \& Sornette (2000)}}
\newcommand{\refTH}{\href{https://doi.org/10.1016/j.jempfin.2018.08.004}{Trimborn \& Härdle (2018)}}
\newcommand{\refHHR}{\href{https://doi.org/10.1093/jjfinec/nbz033}{Härdle, Harvey \& Reule (2020)}}
\newcommand{\refFHH}{\href{https://doi.org/10.1007/978-3-030-13751-9}{Franke, Härdle \& Hafner (2019)}}
\newcommand{\refKupiec}{\href{https://doi.org/10.3905/jod.1995.407942}{Kupiec (1995)}}
\newcommand{\refWang}{\href{https://doi.org/10.1038/s41586-023-06221-2}{Wang et al.\ (2023)}}
\newcommand{\refPeleA}{\href{https://doi.org/10.1080/1351847X.2021.1960403}{Pele et al.\ (2023)}}
\newcommand{\refPeleB}{\href{https://doi.org/10.3390/e21020102}{Pele \& Mazurencu-Marinescu-Pele (2019)}}
\newcommand{\refMiCA}{\href{https://eur-lex.europa.eu/eli/reg/2023/1114/oj}{⟦Regulation||Regulamentul⟧ (EU) 2023/1114 (MiCA)}}
\newcommand{\refESMA}{\href{https://www.esma.europa.eu/esmas-activities/digital-finance-and-innovation/markets-crypto-assets-regulation-mica}{ESMA: MiCA}}
\newcommand{\refGENIUS}{\href{https://www.govinfo.gov/app/details/PLAW-119publ27}{GENIUS Act, Public Law 119-27}}
\newcommand{\refSEC}{\href{https://www.sec.gov/newsroom/speeches-statements/gensler-statement-spot-bitcoin-011023}{SEC (2024)}}
\newcommand{\refFed}{\href{https://www.federalreserve.gov/newsevents/pressreleases/monetary20230312b.htm}{⟦Treasury, Federal Reserve and FDIC||Trezoreria SUA, Rezerva Federală și FDIC⟧ (2023)}}
\newcommand{\refFDIC}{\href{https://www.fdic.gov/news/press-releases/2023/pr23016.html}{FDIC (2023)}}
\newcommand{\refDL}{\href{https://api-docs.defillama.com/}{DefiLlama}}
"""

_BIB = {
    'BHL': r"Baur, D.G., Hong, K., Lee, A.D. (2018). \href{https://doi.org/10.1016/j.intfin.2017.12.004}{Bitcoin: medium of exchange or speculative assets?} \textit{Journal of International Financial Markets, Institutions and Money}, 54, 177--189.",
    'BL': r"Baur, D.G., Lucey, B.M. (2010). \href{https://doi.org/10.1111/j.1540-6288.2010.00244.x}{Is gold a hedge or a safe haven? An analysis of stocks, bonds and gold}. \textit{Financial Review}, 45(2), 217--229.",
    'Boll': r"Bollerslev, T. (1986). \href{https://doi.org/10.1016/0304-4076(86)90063-1}{Generalized autoregressive conditional heteroskedasticity}. \textit{Journal of Econometrics}, 31(3), 307--327.",
    'Buterin': r"Buterin, V. (2014). \href{https://ethereum.org/en/whitepaper/}{Ethereum: a next-generation smart contract and decentralized application platform}. White paper.",
    'CP': r"Caporale, G.M., Plastun, A. (2019). \href{https://doi.org/10.1016/j.frl.2018.11.012}{The day of the week effect in the cryptocurrency market}. \textit{Finance Research Letters}, 31, 258--269.",
    'CF': r"Cheah, E.-T., Fry, J. (2015). \href{https://doi.org/10.1016/j.econlet.2015.02.029}{Speculative bubbles in Bitcoin markets? An empirical investigation into the fundamental value of Bitcoin}. \textit{Economics Letters}, 130, 32--36.",
    'MiCA': r"European Union (2023). \href{https://eur-lex.europa.eu/eli/reg/2023/1114/oj}{Regulation (EU) 2023/1114 on markets in crypto-assets (MiCA)}. \textit{Official Journal of the European Union}, L 150, 40--205.",
    'Fama': r"Fama, E.F. (1970). \href{https://doi.org/10.2307/2325486}{Efficient capital markets: a review of theory and empirical work}. \textit{Journal of Finance}, 25(2), 383--417.",
    'FHH': r"Franke, J., Härdle, W.K., Hafner, C.M. (2019). \href{https://doi.org/10.1007/978-3-030-13751-9}{\textit{Statistics of Financial Markets: An Introduction}}, 5th ed. Springer.",
    'GHMO': r"Gandal, N., Hamrick, J.T., Moore, T., Oberman, T. (2018). \href{https://doi.org/10.1016/j.jmoneco.2017.12.004}{Price manipulation in the Bitcoin ecosystem}. \textit{Journal of Monetary Economics}, 95, 86--96.",
    'Gorton': r"Gorton, G.B., Klee, E.C., Ross, C.P., Ross, S.Y., Vardoulakis, A.P. (2022). \href{https://doi.org/10.3386/w30796}{Leverage and stablecoin pegs}. NBER Working Paper 30796.",
    'GZ': r"Gorton, G.B., Zhang, J. (2021). \href{https://doi.org/10.2139/ssrn.3888752}{Taming wildcat stablecoins}. SSRN Working Paper 3888752; published in \textit{University of Chicago Law Review}, 90(3), 2023.",
    'GS': r"Griffin, J.M., Shams, A. (2020). \href{https://doi.org/10.1111/jofi.12903}{Is Bitcoin really untethered?} \textit{Journal of Finance}, 75(4), 1913--1964.",
    'HHR': r"Härdle, W.K., Harvey, C.R., Reule, R.C.G. (2020). \href{https://doi.org/10.1093/jjfinec/nbz033}{Understanding cryptocurrencies}. \textit{Journal of Financial Econometrics}, 18(2), 181--208.",
    'Hill': r"Hill, B.M. (1975). \href{https://doi.org/10.1214/aos/1176343247}{A simple general approach to inference about the tail of a distribution}. \textit{Annals of Statistics}, 3(5), 1163--1174.",
    'JLS': r"Johansen, A., Ledoit, O., Sornette, D. (2000). \href{https://doi.org/10.1142/S0219024900000115}{Crashes as critical points}. \textit{International Journal of Theoretical and Applied Finance}, 3(2), 219--255.",
    'Katsiampa': r"Katsiampa, P. (2017). \href{https://doi.org/10.1016/j.econlet.2017.06.023}{Volatility estimation for Bitcoin: a comparison of GARCH models}. \textit{Economics Letters}, 158, 3--6.",
    'Kupiec': r"Kupiec, P.H. (1995). \href{https://doi.org/10.3905/jod.1995.407942}{Techniques for verifying the accuracy of risk measurement models}. \textit{Journal of Derivatives}, 3(2), 73--84.",
    'LMS': r"Liu, J., Makarov, I., Schoar, A. (2023). \href{https://doi.org/10.3386/w31160}{Anatomy of a run: the Terra Luna crash}. NBER Working Paper 31160.",
    'LT': r"Liu, Y., Tsyvinski, A. (2021). \href{https://doi.org/10.1093/rfs/hhaa113}{Risks and returns of cryptocurrency}. \textit{Review of Financial Studies}, 34(6), 2689--2727.",
    'LTW': r"Liu, Y., Tsyvinski, A., Wu, X. (2022). \href{https://doi.org/10.1111/jofi.13119}{Common risk factors in cryptocurrency}. \textit{Journal of Finance}, 77(2), 1133--1177.",
    'LM': r"Lo, A.W., MacKinlay, A.C. (1988). \href{https://doi.org/10.1093/rfs/1.1.41}{Stock market prices do not follow random walks: evidence from a simple specification test}. \textit{Review of Financial Studies}, 1(1), 41--66.",
    'LVN': r"Lyons, R.K., Viswanath-Natraj, G. (2023). \href{https://doi.org/10.1016/j.jimonfin.2022.102777}{What keeps stablecoins stable?} \textit{Journal of International Money and Finance}, 131, 102777.",
    'MZZ': r"Ma, Y., Zeng, Y., Zhang, A.L. (2025). \href{https://doi.org/10.3386/w33882}{Stablecoin runs and the centralization of arbitrage}. NBER Working Paper 33882.",
    'MS': r"Makarov, I., Schoar, A. (2020). \href{https://doi.org/10.1016/j.jfineco.2019.07.001}{Trading and arbitrage in cryptocurrency markets}. \textit{Journal of Financial Economics}, 135(2), 293--319.",
    'NC': r"Nadarajah, S., Chu, J. (2017). \href{https://doi.org/10.1016/j.econlet.2016.10.033}{On the inefficiency of Bitcoin}. \textit{Economics Letters}, 150, 6--9.",
    'Nakamoto': r"Nakamoto, S. (2008). \href{https://bitcoin.org/bitcoin.pdf}{Bitcoin: a peer-to-peer electronic cash system}. White paper.",
    'PeleA': r"Pele, D.T., Wesselhöfft, N., Härdle, W.K., Kolossiatis, M., Yatracos, Y.G. (2023). \href{https://doi.org/10.1080/1351847X.2021.1960403}{Are cryptos becoming alternative assets?} \textit{European Journal of Finance}, 29(10), 1064--1105.",
    'PeleB': r"Pele, D.T., Mazurencu-Marinescu-Pele, M. (2019). \href{https://doi.org/10.3390/e21020102}{Using high-frequency entropy to forecast Bitcoin's daily value at risk}. \textit{Entropy}, 21(2), 102.",
    'SEC': r"SEC, U.S. Securities and Exchange Commission (2024). \href{https://www.sec.gov/newsroom/speeches-statements/gensler-statement-spot-bitcoin-011023}{Statement on the approval of spot Bitcoin exchange-traded products}, 10 January 2024.",
    'TL': r"Tran, V.L., Leirvik, T. (2020). \href{https://doi.org/10.1016/j.frl.2019.101382}{Efficiency in the markets of crypto-currencies}. \textit{Finance Research Letters}, 35, 101382.",
    'Fed': r"Treasury, Federal Reserve and FDIC (2023). \href{https://www.federalreserve.gov/newsevents/pressreleases/monetary20230312b.htm}{Joint statement by the Department of the Treasury, Federal Reserve, and FDIC}, 12 March 2023.",
    'TH': r"Trimborn, S., Härdle, W.K. (2018). \href{https://doi.org/10.1016/j.jempfin.2018.08.004}{CRIX an index for cryptocurrencies}. \textit{Journal of Empirical Finance}, 49, 107--122.",
    'GENIUS': r"United States Congress (2025). \href{https://www.govinfo.gov/app/details/PLAW-119publ27}{Guiding and Establishing National Innovation for U.S. Stablecoins Act (GENIUS Act)}, Public Law 119-27.",
    'Urquhart': r"Urquhart, A. (2016). \href{https://doi.org/10.1016/j.econlet.2016.09.019}{The inefficiency of Bitcoin}. \textit{Economics Letters}, 148, 80--82.",
    'Wang': r"Wang, H., Fu, T., Du, Y., Gao, W., et al. (2023). \href{https://doi.org/10.1038/s41586-023-06221-2}{Scientific discovery in the age of artificial intelligence}. \textit{Nature}, 620, 47--60.",
}


def bib(keys=None):
    """Bibliography entries (all, or the given keys), in alphabetical order."""
    ks = keys or list(_BIB)
    return sorted((_BIB[k] for k in ks), key=lambda s: s.lower().replace('{', '').replace('\\', ''))


BIB = bib()
