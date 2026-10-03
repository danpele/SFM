r"""
ch11_common.py -- shared helpers of the Chapter 11 generators (lecture and seminar), SFM
=======================================================================================
Numbers from Quantlets/Ch_11/ch11_numbers.json (generate_fmh_charts.py) and sem11_results.json (seminar11.py);
the clickable citations of Chapter 11 (DOIs checked against Crossref, 3 October 2026).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sfm_build import ROOT, Values   # noqa: E402
from ch1_common import T, date, put_date   # noqa: E402,F401

QL = os.path.join(ROOT, 'Quantlets', 'Ch_11')
QLURL = 'https://github.com/danpele/SFM/tree/main/Quantlets/Ch_11'
NAMES = {'sp500': 'S\\&P 500', 'dax': 'DAX', 'bet': 'BET', 'btc': 'Bitcoin', 'tlv': 'Banca Transilvania (TLV)',
         'snp': 'OMV Petrom (SNP)'}
SHORT = dict(NAMES, tlv='TLV', snp='SNP')
ASSETS = ['sp500', 'dax', 'bet', 'btc', 'tlv', 'snp']


def load():
    with open(os.path.join(QL, 'ch11_numbers.json')) as f:
        return json.load(f)


def load_sem():
    with open(os.path.join(QL, 'sem11_results.json')) as f:
        return json.load(f)


def pct(x, d=0):
    return 100 * x


def values(N):
    """All lecture numbers as @{key} values."""
    V = Values()
    put_date(V, 'end', N['end'])
    E = N['rs_example']
    V.put('ex.mean', E['mean'], 3)
    for i, (dv, y) in enumerate(zip(E['dev'], E['Y']), 1):
        V.put(f'ex.dev{i}', dv, 3, sign=True)
        V.put(f'ex.y{i}', y + 0.0, 3)
    V.put('ex.max', E['max'], 3)
    V.put('ex.min', E['min'], 3)
    V.raw('ex.kmax', str(E['kmax']))
    V.raw('ex.kmin', str(E['kmin']))
    V.put('ex.R', E['R'], 3)
    V.put('ex.S', E['S'], 3)
    V.put('ex.RS', E['RS'], 3)
    V.put('ex.ss', E['sumsq'], 3)
    F = N['fbm']
    for h in ['0.3', '0.5', '0.7']:
        k = h.replace('.', '')
        V.put(f'fbm.{k}.a1', F[h]['acf1'], 2)
        V.put(f'fbm.{k}.a1th', F[h]['acf1_th'], 2)
    G = N['fgn']
    V.put('fgn.8.r1', G['0.8']['rho1'], 3)
    V.put('fgn.8.r10', G['0.8']['rho10'], 3)
    V.put('fgn.8.r50', G['0.8']['rho50'], 3)
    V.put('fgn.8.r100', G['fgn_rho100'], 3)
    V.put('fgn.2.r1', G['0.2']['rho1'], 3)
    V.put('fgn.2.r10', G['0.2']['rho10'], 4)
    V.put('ar.phi', G['ar1_phi'], 3)
    V.put('ar.r10', G['ar1_rho10'], 4)
    A = N['arfima']
    for d, k in [('0.4', 'p'), ('-0.4', 'n')]:
        for c in ['rho1', 'rho2', 'rho10', 'rho50']:
            V.put(f'af.{k}.{c}', A[d][c], 3)
    NI = N['nile']
    V.put('nile.H', NI['H_raw'], 2)
    V.put('nile.Ha', NI['H_adj'], 2)
    V.put('nile.D', NI['dfa_raw'], 2)
    V.put('nile.Da', NI['dfa_adj'], 2)
    V.put('nile.m0', NI['mean_pre'], 0)
    V.put('nile.m1', NI['mean_post'], 0)
    V.put('nile.drop', 100 * (1 - NI['mean_post'] / NI['mean_pre']), 0)
    V.put('nile.q025', NI['q025'], 2)
    V.put('nile.q975', NI['q975'], 2)
    V.put('nile.nm', NI['null_mean'], 2)
    SC = N['scaling']
    for k in ['sp500', 'bet', 'btc']:
        V.put(f'sc.{k}.H', SC[k]['H'], 2)
        V.put(f'sc.{k}.r10', SC[k]['ratio10'], 2)
        V.put(f'sc.{k}.r250', SC[k]['ratio250'], 1)
    V.put('sc.sqrt250', 250 ** 0.5, 1)
    RD = N['rs_dfa']
    for c in ['rs_r', 'rs_abs', 'dfa_r', 'dfa_abs', 'rs_iid', 'dfa_iid']:
        V.put(f'rd.{c}', RD[c], 2)
    V.put('rd.rs10', RD['rs_n10'], 2)
    V.raw('rd.nmax', str(RD['nmax']))
    V.raw('rd.ns', str(RD['nsizes']))
    GP = N['gph']
    for c in ['r', 'abs']:
        V.put(f'gph.{c}.d', GP[c]['d'], 3)
        V.put(f'gph.{c}.se', GP[c]['se'], 3)
        V.put(f'gph.{c}.H', GP[c]['H'], 2)
        V.raw(f'gph.{c}.m', str(GP[c]['m']))
    VA = N['vol_acf']
    for nm, k in [('returns', 'r'), ('absolute', 'a'), ('squared', 's')]:
        for lag in ['1', '10', '50', '100', '250']:
            V.put(f'va.{k}.{lag}', VA[nm][lag], 3)
        V.raw(f'va.{k}.npos', str(VA[nm]['npos']))
    V.put('va.band', VA['band'], 3)
    V.put('va.shuf', VA['abs_shuffled_dfa'], 2)
    V.int('va.n', VA['n'])
    for n_, d in N['mc'].items():
        for m in ['rs', 'dfa', 'gph']:
            V.put(f'mc.{n_}.{m}.mean', d[m]['mean'], 3)
            V.put(f'mc.{n_}.{m}.lo', d[m]['q025'], 2)
            V.put(f'mc.{n_}.{m}.hi', d[m]['q975'], 2)
            V.put(f'mc.{n_}.{m}.w', d[m]['q975'] - d[m]['q025'], 2)
    L = N['lo']
    for phi, d in L['ar'].items():
        k = phi.replace('.', '')
        V.put(f'lo.ar{k}.c', 100 * d['classical'], 0)
        V.put(f'lo.ar{k}.m', 100 * d['modified'], 0)
        V.put(f'lo.ar{k}.q', d['q_mean'], 1)
    for h, d in L['fgn'].items():
        k = h.replace('.', '')
        V.put(f'lo.h{k}.c', 100 * d['classical'], 0)
        V.put(f'lo.h{k}.m', 100 * d['modified'], 0)
    SP = N['spurious']
    for dl, d in SP['break'].items():
        k = dl.replace('.', '')
        for m in ['rs', 'dfa', 'gph']:
            V.put(f'br.{k}.{m}', d[m], 2)
    for p, d in SP['garch'].items():
        k = p.replace('.', '')
        V.put(f'ga.{k}.adfa', d['abs_dfa'], 2)
        V.put(f'ga.{k}.agph', d['abs_gph'], 2)
        V.put(f'ga.{k}.rdfa', d['ret_dfa'], 2)
    V.raw('sp.n', str(SP['N']))
    V.raw('sp.reps', str(SP['reps']))
    M = N['markets']
    for k, d in M.items():
        V.int(f'm.{k}.n', d['r']['n'])
        V.raw(f'm.{k}.y0', d['first'][:4])
        for part in ['r', 'abs']:
            e = d[part]
            V.put(f'm.{k}.{part}.rs', e['rs'], 2)
            V.put(f'm.{k}.{part}.dfa', e['dfa'], 2)
            V.put(f'm.{k}.{part}.d', e['gph_d'], 2)
            V.put(f'm.{k}.{part}.dse', e['gph_se'], 2)
            V.put(f'm.{k}.{part}.V', e['lo_V'], 2)
            V.raw(f'm.{k}.{part}.q', str(e['lo_q']))
            V.put(f'm.{k}.{part}.V0', e['lo_V0'], 2)
        for m in ['rs', 'dfa', 'gph']:
            V.put(f'm.{k}.mc.{m}.lo', d['mc'][m]['q025'], 2)
            V.put(f'm.{k}.mc.{m}.hi', d['mc'][m]['q975'], 2)
            V.put(f'm.{k}.mc.{m}.mean', d['mc'][m]['mean'], 2)
        V.put(f'm.{k}.shuf', d['abs_shuffled_dfa'], 2)
    RO = N['rolling']
    V.put('rl.lo', RO['band']['q025'], 2)
    V.put('rl.hi', RO['band']['q975'], 2)
    for k in ['sp500', 'bet', 'btc']:
        d = RO[k]
        V.raw(f'rl.{k}.nw', str(d['n_windows']))
        for c in ['min', 'max', 'last', 'abs_min', 'abs_max', 'abs_last', 'first_third', 'last_third']:
            V.put(f'rl.{k}.{c}', d[c], 2)
        V.put(f'rl.{k}.above', 100 * d['above'], 0)
        V.put(f'rl.{k}.below', 100 * d['below'], 0)
        put_date(V, f'rl.{k}.dmax', d['date_max'])
        put_date(V, f'rl.{k}.dmin', d['date_min'])
        put_date(V, f'rl.{k}.e0', d['first_end'])
        V.put(f'rl.{k}.fa', d['n_windows'] * 0.05, 0)
    return V


# -----------------------------------------------------------------------------
# Clickable citations (DOIs verified via Crossref, 3 October 2026)
# -----------------------------------------------------------------------------
REFS = r"""
\newcommand{\refPeters}{\href{https://www.wiley.com/en-us/Fractal+Market+Analysis\%3A+Applying+Chaos+Theory+to+Investment+and+Economics-p-9780471585244}{Peters (1994)}}
\newcommand{\refFama}{\href{https://doi.org/10.2307/2325486}{Fama (1970)}}
\newcommand{\refLoAMH}{\href{https://doi.org/10.3905/jpm.2004.442611}{Lo (2004)}}
\newcommand{\refMan}{\href{https://doi.org/10.1086/294632}{Mandelbrot (1963)}}
\newcommand{\refCoast}{\href{https://doi.org/10.1126/science.156.3775.636}{Mandelbrot (1967)}}
\newcommand{\refMVN}{\href{https://doi.org/10.1137/1010093}{Mandelbrot \& Van Ness (1968)}}
\newcommand{\refNoah}{\href{https://doi.org/10.1029/WR004i005p00909}{Mandelbrot \& Wallis (1968)}}
\newcommand{\refMW}{\href{https://doi.org/10.1029/WR005i005p00967}{Mandelbrot \& Wallis (1969)}}
\newcommand{\refHurst}{\href{https://doi.org/10.1061/TACEAT.0006518}{Hurst (1951)}}
\newcommand{\refAL}{\href{https://doi.org/10.1093/biomet/63.1.111}{Anis \& Lloyd (1976)}}
\newcommand{\refLo}{\href{https://doi.org/10.2307/2938368}{Lo (1991)}}
\newcommand{\refNW}{\href{https://doi.org/10.2307/1913610}{Newey \& West (1987)}}
\newcommand{\refTTW}{\href{https://doi.org/10.1016/S0378-3758(98)00250-X}{Teverovsky, Taqqu \& Willinger (1999)}}
\newcommand{\refPeng}{\href{https://doi.org/10.1103/PhysRevE.49.1685}{Peng et al.\ (1994)}}
\newcommand{\refKan}{\href{https://doi.org/10.1016/S0378-4371(01)00144-3}{Kantelhardt et al.\ (2001)}}
\newcommand{\refGPH}{\href{https://doi.org/10.1111/j.1467-9892.1983.tb00371.x}{Geweke \& Porter-Hudak (1983)}}
\newcommand{\refRob}{\href{https://doi.org/10.1214/aos/1176324317}{Robinson (1995)}}
\newcommand{\refGJ}{\href{https://doi.org/10.1111/j.1467-9892.1980.tb00297.x}{Granger \& Joyeux (1980)}}
\newcommand{\refHosking}{\href{https://doi.org/10.1093/biomet/68.1.165}{Hosking (1981)}}
\newcommand{\refBaillie}{\href{https://doi.org/10.1016/0304-4076(95)01732-1}{Baillie (1996)}}
\newcommand{\refDGE}{\href{https://doi.org/10.1016/0927-5398(93)90006-D}{Ding, Granger \& Engle (1993)}}
\newcommand{\refDI}{\href{https://doi.org/10.1016/S0304-4076(01)00073-2}{Diebold \& Inoue (2001)}}
\newcommand{\refGH}{\href{https://doi.org/10.1016/j.jempfin.2003.03.001}{Granger \& Hyung (2004)}}
\newcommand{\refLS}{\href{https://doi.org/10.1080/07350015.1998.10524760}{Lobato \& Savin (1998)}}
\newcommand{\refCobb}{\href{https://doi.org/10.1093/biomet/65.2.243}{Cobb (1978)}}
\newcommand{\refBBM}{\href{https://doi.org/10.1016/S0304-4076(95)01749-6}{Baillie, Bollerslev \& Mikkelsen (1996)}}
\newcommand{\refAB}{\href{https://doi.org/10.1111/j.1540-6261.1997.tb02722.x}{Andersen \& Bollerslev (1997)}}
\newcommand{\refMuller}{\href{https://doi.org/10.1016/S0927-5398(97)00007-8}{Müller et al.\ (1997)}}
\newcommand{\refCorsi}{\href{https://doi.org/10.1093/jjfinec/nbp001}{Corsi (2009)}}
\newcommand{\refWeron}{\href{https://doi.org/10.1016/S0378-4371(02)00961-5}{Weron (2002)}}
\newcommand{\refCT}{\href{https://doi.org/10.1016/j.physa.2003.12.031}{Cajueiro \& Tabak (2004)}}
\newcommand{\refUrq}{\href{https://doi.org/10.1016/j.econlet.2016.09.019}{Urquhart (2016)}}
\newcommand{\refNC}{\href{https://doi.org/10.1016/j.econlet.2016.10.033}{Nadarajah \& Chu (2017)}}
\newcommand{\refBar}{\href{https://doi.org/10.1016/j.econlet.2017.09.013}{Bariviera (2017)}}
\newcommand{\refKrisA}{\href{https://doi.org/10.1142/S0219525912500658}{Kristoufek (2012)}}
\newcommand{\refKrisB}{\href{https://doi.org/10.1038/srep02857}{Kristoufek (2013)}}
\newcommand{\refCarlson}{\href{https://www.federalreserve.gov/pubs/feds/2007/200713/index.html}{Carlson (2007)}}
\newcommand{\refRogers}{\href{https://doi.org/10.1111/1467-9965.00025}{Rogers (1997)}}
\newcommand{\refFHH}{\href{https://doi.org/10.1007/978-3-030-13751-9}{Franke, Härdle \& Hafner (2019)}}
\newcommand{\refWang}{\href{https://doi.org/10.1038/s41586-023-06221-2}{Wang et al.\ (2023)}}
\newcommand{\refPeleA}{\href{https://doi.org/10.1016/j.sbspro.2012.09.1030}{Pele \& Mazurencu-Marinescu (2012)}}
\newcommand{\refPeleB}{\href{https://doi.org/10.5018/economics-ejournal.ja.2019-29}{Pele \& Mazurencu-Marinescu-Pele (2019)}}
"""

_BIB = {
    'AL': r"Anis, A.A., Lloyd, E.H. (1976). \href{https://doi.org/10.1093/biomet/63.1.111}{The expected value of the adjusted rescaled Hurst range of independent normal summands}. \textit{Biometrika}, 63(1), 111--116.",
    'AB': r"Andersen, T.G., Bollerslev, T. (1997). \href{https://doi.org/10.1111/j.1540-6261.1997.tb02722.x}{Heterogeneous information arrivals and return volatility dynamics: uncovering the long-run in high frequency returns}. \textit{Journal of Finance}, 52(3), 975--1005.",
    'Baillie': r"Baillie, R.T. (1996). \href{https://doi.org/10.1016/0304-4076(95)01732-1}{Long memory processes and fractional integration in econometrics}. \textit{Journal of Econometrics}, 73(1), 5--59.",
    'BBM': r"Baillie, R.T., Bollerslev, T., Mikkelsen, H.O. (1996). \href{https://doi.org/10.1016/S0304-4076(95)01749-6}{Fractionally integrated generalized autoregressive conditional heteroskedasticity}. \textit{Journal of Econometrics}, 74(1), 3--30.",
    'Bar': r"Bariviera, A.F. (2017). \href{https://doi.org/10.1016/j.econlet.2017.09.013}{The inefficiency of Bitcoin revisited: a dynamic approach}. \textit{Economics Letters}, 161, 1--4.",
    'CT': r"Cajueiro, D.O., Tabak, B.M. (2004). \href{https://doi.org/10.1016/j.physa.2003.12.031}{The Hurst exponent over time: testing the assertion that emerging markets are becoming more efficient}. \textit{Physica A}, 336(3--4), 521--537.",
    'Carlson': r"Carlson, M. (2007). \href{https://www.federalreserve.gov/pubs/feds/2007/200713/index.html}{A brief history of the 1987 stock market crash with a discussion of the Federal Reserve response}. Finance and Economics Discussion Series 2007-13, Federal Reserve Board.",
    'Cobb': r"Cobb, G.W. (1978). \href{https://doi.org/10.1093/biomet/65.2.243}{The problem of the Nile: conditional solution to a changepoint problem}. \textit{Biometrika}, 65(2), 243--251.",
    'Corsi': r"Corsi, F. (2009). \href{https://doi.org/10.1093/jjfinec/nbp001}{A simple approximate long-memory model of realized volatility}. \textit{Journal of Financial Econometrics}, 7(2), 174--196.",
    'DI': r"Diebold, F.X., Inoue, A. (2001). \href{https://doi.org/10.1016/S0304-4076(01)00073-2}{Long memory and regime switching}. \textit{Journal of Econometrics}, 105(1), 131--159.",
    'DGE': r"Ding, Z., Granger, C.W.J., Engle, R.F. (1993). \href{https://doi.org/10.1016/0927-5398(93)90006-D}{A long memory property of stock market returns and a new model}. \textit{Journal of Empirical Finance}, 1(1), 83--106.",
    'Fama': r"Fama, E.F. (1970). \href{https://doi.org/10.2307/2325486}{Efficient capital markets: a review of theory and empirical work}. \textit{Journal of Finance}, 25(2), 383--417.",
    'FHH': r"Franke, J., Härdle, W.K., Hafner, C.M. (2019). \href{https://doi.org/10.1007/978-3-030-13751-9}{\textit{Statistics of Financial Markets: An Introduction}}, 5th ed. Springer.",
    'GPH': r"Geweke, J., Porter-Hudak, S. (1983). \href{https://doi.org/10.1111/j.1467-9892.1983.tb00371.x}{The estimation and application of long memory time series models}. \textit{Journal of Time Series Analysis}, 4(4), 221--238.",
    'GH': r"Granger, C.W.J., Hyung, N. (2004). \href{https://doi.org/10.1016/j.jempfin.2003.03.001}{Occasional structural breaks and long memory with an application to the S\&P 500 absolute stock returns}. \textit{Journal of Empirical Finance}, 11(3), 399--421.",
    'GJ': r"Granger, C.W.J., Joyeux, R. (1980). \href{https://doi.org/10.1111/j.1467-9892.1980.tb00297.x}{An introduction to long-memory time series models and fractional differencing}. \textit{Journal of Time Series Analysis}, 1(1), 15--29.",
    'Hosking': r"Hosking, J.R.M. (1981). \href{https://doi.org/10.1093/biomet/68.1.165}{Fractional differencing}. \textit{Biometrika}, 68(1), 165--176.",
    'Hurst': r"Hurst, H.E. (1951). \href{https://doi.org/10.1061/TACEAT.0006518}{Long-term storage capacity of reservoirs}. \textit{Transactions of the American Society of Civil Engineers}, 116(1), 770--799.",
    'Kan': r"Kantelhardt, J.W., Koscielny-Bunde, E., Rego, H.H.A., Havlin, S., Bunde, A. (2001). \href{https://doi.org/10.1016/S0378-4371(01)00144-3}{Detecting long-range correlations with detrended fluctuation analysis}. \textit{Physica A}, 295(3--4), 441--454.",
    'KrisA': r"Kristoufek, L. (2012). \href{https://doi.org/10.1142/S0219525912500658}{Fractal markets hypothesis and the global financial crisis: scaling, investment horizons and liquidity}. \textit{Advances in Complex Systems}, 15(6), 1250065.",
    'KrisB': r"Kristoufek, L. (2013). \href{https://doi.org/10.1038/srep02857}{Fractal markets hypothesis and the global financial crisis: wavelet power evidence}. \textit{Scientific Reports}, 3, 2857.",
    'Lo': r"Lo, A.W. (1991). \href{https://doi.org/10.2307/2938368}{Long-term memory in stock market prices}. \textit{Econometrica}, 59(5), 1279--1313.",
    'LoAMH': r"Lo, A.W. (2004). \href{https://doi.org/10.3905/jpm.2004.442611}{The adaptive markets hypothesis}. \textit{Journal of Portfolio Management}, 30(5), 15--29.",
    'LS': r"Lobato, I.N., Savin, N.E. (1998). \href{https://doi.org/10.1080/07350015.1998.10524760}{Real and spurious long-memory properties of stock-market data}. \textit{Journal of Business \& Economic Statistics}, 16(3), 261--268.",
    'Man': r"Mandelbrot, B. (1963). \href{https://doi.org/10.1086/294632}{The variation of certain speculative prices}. \textit{Journal of Business}, 36(4), 394--419.",
    'Coast': r"Mandelbrot, B. (1967). \href{https://doi.org/10.1126/science.156.3775.636}{How long is the coast of Britain? Statistical self-similarity and fractional dimension}. \textit{Science}, 156(3775), 636--638.",
    'MVN': r"Mandelbrot, B.B., Van Ness, J.W. (1968). \href{https://doi.org/10.1137/1010093}{Fractional Brownian motions, fractional noises and applications}. \textit{SIAM Review}, 10(4), 422--437.",
    'Noah': r"Mandelbrot, B.B., Wallis, J.R. (1968). \href{https://doi.org/10.1029/WR004i005p00909}{Noah, Joseph, and operational hydrology}. \textit{Water Resources Research}, 4(5), 909--918.",
    'MW': r"Mandelbrot, B.B., Wallis, J.R. (1969). \href{https://doi.org/10.1029/WR005i005p00967}{Robustness of the rescaled range R/S in the measurement of noncyclic long run statistical dependence}. \textit{Water Resources Research}, 5(5), 967--988.",
    'Muller': r"Müller, U.A., Dacorogna, M.M., Davé, R.D., Olsen, R.B., Pictet, O.V., von Weizsäcker, J.E. (1997). \href{https://doi.org/10.1016/S0927-5398(97)00007-8}{Volatilities of different time resolutions: analyzing the dynamics of market components}. \textit{Journal of Empirical Finance}, 4(2--3), 213--239.",
    'NC': r"Nadarajah, S., Chu, J. (2017). \href{https://doi.org/10.1016/j.econlet.2016.10.033}{On the inefficiency of Bitcoin}. \textit{Economics Letters}, 150, 6--9.",
    'NW': r"Newey, W.K., West, K.D. (1987). \href{https://doi.org/10.2307/1913610}{A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix}. \textit{Econometrica}, 55(3), 703--708.",
    'PeleA': r"Pele, D.T., Mazurencu-Marinescu, M. (2012). \href{https://doi.org/10.1016/j.sbspro.2012.09.1030}{Modelling stock market crashes: the case of Bucharest Stock Exchange}. \textit{Procedia -- Social and Behavioral Sciences}, 58, 533--542.",
    'PeleB': r"Pele, D.T., Mazurencu-Marinescu-Pele, M. (2019). \href{https://doi.org/10.5018/economics-ejournal.ja.2019-29}{Metcalfe's law and log-period power laws in the cryptocurrencies market}. \textit{Economics}, 13(1), 20190029.",
    'Peng': r"Peng, C.-K., Buldyrev, S.V., Havlin, S., Simons, M., Stanley, H.E., Goldberger, A.L. (1994). \href{https://doi.org/10.1103/PhysRevE.49.1685}{Mosaic organization of DNA nucleotides}. \textit{Physical Review E}, 49(2), 1685--1689.",
    'Peters': r"Peters, E.E. (1994). \href{https://www.wiley.com/en-us/Fractal+Market+Analysis\%3A+Applying+Chaos+Theory+to+Investment+and+Economics-p-9780471585244}{\textit{Fractal Market Analysis: Applying Chaos Theory to Investment and Economics}}. Wiley.",
    'Rob': r"Robinson, P.M. (1995). \href{https://doi.org/10.1214/aos/1176324317}{Gaussian semiparametric estimation of long range dependence}. \textit{Annals of Statistics}, 23(5), 1630--1661.",
    'Rogers': r"Rogers, L.C.G. (1997). \href{https://doi.org/10.1111/1467-9965.00025}{Arbitrage with fractional Brownian motion}. \textit{Mathematical Finance}, 7(1), 95--105.",
    'TTW': r"Teverovsky, V., Taqqu, M.S., Willinger, W. (1999). \href{https://doi.org/10.1016/S0378-3758(98)00250-X}{A critical look at Lo's modified R/S statistic}. \textit{Journal of Statistical Planning and Inference}, 80(1--2), 211--227.",
    'Urq': r"Urquhart, A. (2016). \href{https://doi.org/10.1016/j.econlet.2016.09.019}{The inefficiency of Bitcoin}. \textit{Economics Letters}, 148, 80--82.",
    'Wang': r"Wang, H., Fu, T., Du, Y., Gao, W., et al. (2023). \href{https://doi.org/10.1038/s41586-023-06221-2}{Scientific discovery in the age of artificial intelligence}. \textit{Nature}, 620, 47--60.",
    'Weron': r"Weron, R. (2002). \href{https://doi.org/10.1016/S0378-4371(02)00961-5}{Estimating long-range dependence: finite sample properties and confidence intervals}. \textit{Physica A}, 312(1--2), 285--299.",
}


def bib(keys=None):
    """Bibliography entries (all, or the given keys), in alphabetical order."""
    ks = keys or list(_BIB)
    return sorted((_BIB[k] for k in ks), key=lambda s: s.lower().replace('{', '').replace('\\', ''))


BIB = bib()
