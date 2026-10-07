r"""
build_chapter2.py -- Capitolul 2 (Distribuții clasice și fapte stilizate), EN + RO dintr-o singură sursă
======================================================================================================
Text ⟦english||română⟧; cifrele @{cheie} vin din Quantlets/Ch_02/ch2_numbers.json (generate_all_charts.py),
din sem2_results.json (seminar2.py) sau sînt calculate aici, în Python, pentru exemplele lucrate.
Nicio cifră nu este scrisă de mînă.
Ieșire:
  EN/Courses/chapter2_classical_distributions_stylised_facts.tex
  RO/Cursuri/capitol2_distributii_clasice_fapte_stilizate.tex
Rulare:
  python3 Quantlets/Ch_02/generate_all_charts.py && python3 Quantlets/Ch_02/seminar2.py
  python3 latex/build_chapter2.py && python3 latex/sfm_build.py compile 2
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy import stats   # noqa: E402

from sfm_build import Deck, cols, items, table, photo, n as num   # noqa: E402
from ch2_common import (ASSETS, BIB, INDICES, NAMES, REFS, SHORTNAME, T, big, load, load_sem, put_date, sci,   # noqa: E402
                        values)

N = load()
S = load_sem()
V = values(N)
D = Deck(2, 'lecture', refs=REFS)
C = 'https://commons.wikimedia.org/wiki/File:'


def ql(folder):
    return f'\\sfmquantlet{{Ch_02}}{{{folder}}}'


def chart(title, fig, folder, bullets, h='0.60\\textheight', size='footnotesize'):
    body = (f'\\begin{{center}}\n\\includegraphics[width=0.97\\textwidth,height={h},keepaspectratio]{{{fig}.pdf}}\n'
            f'\\end{{center}}\n\\vspace{{-0.25cm}}\n' + items(*bullets) + '\n' + ql(folder))
    D.frame(title, body, size)


def side(title, fig, folder, bullets, w=0.56, size='footnotesize', h='0.78\\textheight'):
    left = f'\\begin{{center}}\n\\includegraphics[width=\\linewidth,height={h},keepaspectratio]{{{fig}.pdf}}\n\\end{{center}}'
    D.frame(title, cols(left, items(*bullets), wl=f'{w:.2f}', wr=f'{0.96 - w:.2f}') + '\n' + ql(folder), size)


PH = {
    'gauss': ('ch2_gauss_1840.jpg', C + 'Carl_Friedrich_Gauss_1840_by_Jensen.jpg',
              '⟦Portrait||Portret⟧: Christian Albrecht Jensen (1840); ⟦public domain||domeniu public⟧; Wikimedia Commons'),
    'galton': ('ch2_galton_board.jpg', C + 'Galton_box_2.jpg',
               '⟦Photo||Foto⟧: Klaus-Dieter Keller (2010); ⟦public domain||domeniu public⟧; Wikimedia Commons'),
    'gosset': ('ch2_gosset_1908.jpg', C + 'William_Sealy_Gosset.jpg',
               '⟦Photo||Foto⟧: ⟦unknown author||autor necunoscut⟧ (1908); ⟦public domain||domeniu public⟧; Wikimedia Commons'),
    'cont': ('ch2_cont_2012.jpg', C + 'Rama_Cont_Oberwolfach_2012.jpg',
             '⟦Photo||Foto⟧: Renate Schmid (2012); CC BY-SA 2.0 DE; Wikimedia Commons'),
    'mandelbrot': ('ch0_mandelbrot_2007.jpg', C + 'Benoit_Mandelbrot_mg_1804-d.jpg',
                   '⟦Photo||Foto⟧: Rama (2007); CC BY-SA 2.0 fr; Wikimedia Commons'),
}


def ph(key, cap, h='0.5\\textheight'):
    f, url, cred = PH[key]
    return photo(f, cap, url, cred, h=h)


# =============================================================================
# CIFRE CALCULATE AICI
# =============================================================================
M = N['mom']
# tabelul cozilor distribuției Normale
for k, d in N['normal_tails'].items():
    V.raw(f'nt.{k}.p', sci(d['p'], 1) if d['p'] < 0.01 else num(d['p'], 4))
    V.raw(f'nt.{k}.days', big(d['every_days']))
    V.raw(f'nt.{k}.years', big(d['every_years']) if d['every_years'] >= 10 else num(d['every_years'], 1 if d['every_years'] >= 1 else 3))
V.put('nt.1.in', 100 * (1 - N['normal_tails']['1']['p']), 2)
V.put('nt.2.in', 100 * (1 - N['normal_tails']['2']['p']), 2)
V.put('nt.3.in', 100 * (1 - N['normal_tails']['3']['p']), 2)
# cea mai proastă zi a S&P 500
sp = M['sp500']
zmin = (sp['min'] - sp['mean']) / sp['sd']
V.raw('sp.pmin', sci(stats.norm.cdf(zmin), 0))
V.raw('sp.pmin.exp', str(int(math.floor(math.log10(1 / stats.norm.cdf(zmin) / 252)))))
# zile cu pierderi > 3% și > 5%
L = N['loss']
for k in ['sp500', 'bet', 'btc']:
    for thr in ['3', '5', '10']:
        V.raw(f'loss.{k}.{thr}.obs', str(L[k][thr]['obs']))
        d = L[k][thr]['exp']
        V.raw(f'loss.{k}.{thr}.exp', num(d, 1) if d >= 0.05 else sci(d, 1))
        V.put(f'loss.{k}.{thr}.z', L[k][thr]['z'], 2)
V.put('sp.p3', stats.norm.cdf(L['sp500']['3']['z']) * 100, 2)
# zile de 4 sigma
SG = N['sigma']
for k in ASSETS:
    for j in ['3', '4', '5', '6']:
        V.raw(f'sg.{k}.{j}.obs', str(SG[k][j]['obs']))
        e = SG[k][j]['exp']
        V.raw(f'sg.{k}.{j}.exp', num(e, 2) if e >= 0.01 else sci(e, 1))
        V.raw(f'sg.{k}.{j}.pp', sci(SG[k][j]['poisson_p'], 0))
    V.put(f'sg.{k}.every', SG[k]['4']['every_obs'], 0)
    V.put(f'sg.{k}.peryear', M[k]['ppy'] / SG[k]['4']['every_obs'], 1)
V.raw('sg.normal.every', big(SG['sp500']['4']['every_normal']))
V.put('sg.normal.years', SG['sp500']['4']['every_normal'] / 252, 0)
V.put('sg.sp.ratio', SG['sp500']['4']['obs'] / SG['sp500']['4']['exp'], 0)
# lognormal: BET, un an și zece ani
bet = M['bet']
mu_y = bet['mean'] * bet['ppy'] / 100
s_y = bet['sd'] * math.sqrt(bet['ppy']) / 100
V.put('ln.mu', 100 * mu_y, 1)
V.put('ln.s', 100 * s_y, 1)
for T_ in [1, 10]:
    mT, sT = mu_y * T_, s_y * math.sqrt(T_)
    V.put(f'ln.{T_}.med', 100 * math.exp(mT), 1)
    V.put(f'ln.{T_}.mean', 100 * math.exp(mT + sT ** 2 / 2), 1)
    V.put(f'ln.{T_}.ploss', 100 * stats.norm.cdf(-mT / sT), 1)
    V.put(f'ln.{T_}.q05', 100 * math.exp(mT + sT * stats.norm.ppf(0.05)), 1)
V.put('ln.drag', 100 * s_y ** 2 / 2, 2)
Q05_10 = 100 * math.exp(10 * mu_y + s_y * math.sqrt(10) * stats.norm.ppf(0.05))   # cuantila de 5% după 10 ani
V.put('ln.ppy', bet['ppy'], 0)
LN = N['lognormal']
for s_, key in [('0.25', 'a'), ('0.5', 'b'), ('1.0', 'c')]:
    V.put(f'lnd.{key}.mean', LN[s_]['mean'], 2)
    V.put(f'lnd.{key}.mode', LN[s_]['mode'], 2)
    V.put(f'lnd.{key}.skew', LN[s_]['skew'], 2)
# CLT și aproximarea Normală
CL = N['clt']
for n_ in ['5', '30', '300']:
    V.put(f'clt.{n_}.tail', 100 * CL[n_]['p_tail'], 1)
    V.put(f'clt.{n_}.skew', CL[n_]['skew_mean'], 2)
V.put('clt.ntail', 100 * CL['5']['normal_tail'], 1)
AP = N['approx']
for key in AP:
    V.put(f'ap.{key}.err', AP[key]['max_cdf_err'], 3)
    V.put(f'ap.{key}.v', AP[key]['np1p'], 1)
# densitatea DAX
DX = N['dax']
V.put('dax.in1', 100 * DX['within1'], 1)
V.put('dax.nin1', 100 * DX['normal_within1'], 1)
V.put('dax.peak', DX['peak_kde'], 2)
V.put('dax.npeak', DX['peak_normal'], 2)
V.put('dax.ratio6', DX['ratio_at6'], 0)
# TLV
TL = N['tlv']
V.put('tlv.r1', TL['r_bad'][0], 1)
V.put('tlv.r2', TL['r_bad'][1], 1)
V.put('tlv.k', TL['exkurt'], 1)
V.put('tlv.kc', TL['exkurt_clean'], 1)
V.put('tlv.c0', TL['close']['2016-05-27'], 2)
V.put('tlv.c1', TL['close']['2016-05-30'], 2)
V.put('tlv.a0', TL['adj']['2016-05-27'], 2)
V.put('tlv.a1', TL['adj']['2016-05-30'], 2)
V.put('tlv.a2', TL['adj']['2016-05-31'], 2)
V.put('tlv.nu', TL['nu'], 2)
V.put('tlv.nuc', TL['nu_clean'], 2)
# JB, exemplul lucrat
V.put('jb.s2', sp['skew'] ** 2, 3)
V.put('jb.k24', sp['exkurt'] ** 2 / 4, 1)
V.put('jb.n6', sp['n'] / 6, 1)
V.put('jb.crit', stats.chi2.ppf(0.95, 2), 2)
B1 = S['B1']
V.put('bs.klo', B1['bs_k_lo'], 1)
V.put('bs.khi', B1['bs_k_hi'], 1)
V.put('bs.nlo', B1['n_lo_k'], 2)
V.put('bs.nhi', B1['n_hi_k'], 2)
# QQ
QN = N['qq_n']
for k in ['sp500', 'bet', 'btc']:
    V.put(f'qq.{k}.min', QN[k]['min_emp'], 1)
    V.put(f'qq.{k}.tmin', QN[k]['min_theo'], 1)
QT = N['qq_t']
V.put('qqt.btc.min', QT['btc']['min_emp'], 1)
V.put('qqt.btc.tmin', QT['btc']['min_theo'], 1)
V.put('qqt.sp.min', QT['sp500']['min_emp'], 1)
V.put('qqt.sp.tmin', QT['sp500']['min_theo'], 1)
# Student-t
TD = N['tdens']
for nu in ['3', '5', '10']:
    V.put(f'td.{nu}.p4', 100 * TD[nu]['p4'], 2)
V.put('td.n.p4', 100 * TD['normal_p4'], 4)
V.put('td.ratio5', TD['5']['p4'] / TD['normal_p4'], 0)
B4 = S['B4']
for k in INDICES:
    V.put(f'b4.{k}.expt', B4[k]['exp_t'], 1)
# agregare
AG = N['agg']
for k in INDICES:
    for h in ['1', '5', '21', '63']:
        V.put(f'ag.{k}.{h}', AG[k][h]['exkurt'], 1)
V.raw('ag.btc.63.n', str(AG['btc']['63']['n']))
V.put('ag.btc.63.p', AG['btc']['63']['jb_p'], 2)
V.put('ag.bet.63.p', AG['bet']['63']['jb_p'], 2)
QH = N['qqh']
V.put('qh.21.skew', QH['21']['skew'], 2)
V.put('qh.21.k', QH['21']['exkurt'], 1)
V.raw('qh.21.n', str(QH['21']['n']))
# ACF
AC = N['acf']
for k in ['sp500', 'bet', 'btc']:
    V.put(f'acf.{k}.r1', AC[k]['rho_r'][0], 2)
    V.put(f'acf.{k}.a1', AC[k]['rho_abs'][0], 2)
    V.put(f'acf.{k}.a10', AC[k]['rho_abs'][9], 2)
    V.put(f'acf.{k}.a50', AC[k]['rho_abs'][49], 2)
    V.put(f'acf.{k}.lbr', AC[k]['lb_r']['Q'], 0)
    V.put(f'acf.{k}.lba', AC[k]['lb_abs']['Q'], 0)
    V.put(f'acf.{k}.band', AC[k]['band'], 3)
    V.raw(f'acf.{k}.nout', str(AC[k]['n_out_r']))
V.put('lb.crit', stats.chi2.ppf(0.95, 10), 1)
# leverage
LV = N['lev']
for k in INDICES:
    V.put(f'lev.{k}.m5', LV[k]['mean5'], 2)
    V.put(f'lev.{k}.l1', LV[k]['L'][0], 2)
# gain/loss
GL = N['gl']
for k in ASSETS:
    V.put(f'gl.{k}.q01', -GL[k]['q01'], 2)
    V.put(f'gl.{k}.q99', GL[k]['q99'], 2)
    V.raw(f'gl.{k}.d', str(GL[k]['down4']))
    V.raw(f'gl.{k}.u', str(GL[k]['up4']))
# ani: AI
YR = N['years']
yb = sorted(YR['btc'], key=int)
nub = [YR['btc'][y]['nu'] for y in yb]
rho_b, p_b = stats.spearmanr([int(y) for y in yb], nub)
V.put('yr.rho', rho_b, 2)
V.put('yr.p', p_b, 2)
V.put('yr.numin', min(nub), 1)
V.put('yr.numax', max(nub), 1)
V.raw('yr.n', str(len(yb)))
V.raw('yr.first', yb[0])
V.raw('yr.last', yb[-1])
V.put('yr.k2020', YR['btc']['2020']['exkurt'], 0)
# verificați-vă
V.put('cy.z', (-3 - 0.05) / 1.0, 2)
V.put('cy.jb', 2500 / 6 * (0.5 ** 2 + 6 ** 2 / 4), 0)
V.put('cy.mean', 100 * math.exp(0.05 + 0.2 ** 2 / 2), 1)
V.put('cy.med', 100 * math.exp(0.05), 1)


# extreme și intervale calculate (nicio cifră scrisă de mînă în text)
ku = {k: M[k]['exkurt'] for k in ASSETS}
nus = {k: M[k]['nu'] for k in ASSETS}
V.raw('ku.min.name', SHORTNAME[min(ku, key=ku.get)])
V.raw('ku.max.name', SHORTNAME[max(ku, key=ku.get)])
V.put('ku.min', min(ku.values()), 1)
V.put('ku.max', max(ku.values()), 1)
V.put('ku.min0', math.floor(min(ku.values())), 0)
V.put('ku.max0', math.ceil(max(ku.values())), 0)
V.raw('nu.min.name', SHORTNAME[min(nus, key=nus.get)])
V.raw('nu.max.name', SHORTNAME[max(nus, key=nus.get)])
V.put('nu.min', min(nus.values()), 2)
V.put('nu.max', max(nus.values()), 2)
V.put('nu.min0', math.floor(min(nus.values())), 0)
V.put('nu.max0', math.ceil(max(nus.values())), 0)
V.raw('nu.below4', str(sum(v < 4 for v in nus.values())))
V.raw('nassets', str(len(ASSETS)))
daic = {k: M[k]['aic_n'] - M[k]['aic_t'] for k in ASSETS}
V.put('daic.min', min(daic.values()), 0)
ev = {k: SG[k]['4']['every_obs'] for k in INDICES}
V.put('ev.min', min(ev.values()), 0)
V.put('ev.max', max(ev.values()), 0)
stk = ['snp', 'brd', 'snn', 'sng']
V.raw('stk.min', str(min(SG[k]['4']['obs'] for k in stk)))
V.raw('stk.max', str(max(SG[k]['4']['obs'] for k in stk)))
ex4 = [SG[k]['4']['exp'] for k in ASSETS]
V.put('ex4.min', min(ex4), 2)
V.put('ex4.max', max(ex4), 2)
pk = [N['normal_tails'][str(k)]['p'] for k in range(1, 7)]
ratios = [pk[i] / pk[i + 1] for i in range(5)]
V.put('nt.rmin', min(ratios), 0)
V.put('nt.rmax', max(ratios), 0)
V.put('years.span', (M['sp500']['n'] / M['sp500']['ppy']), 0)
V.int('ag.n1', AG['sp500']['1']['n'])
V.raw('ag.n21', str(AG['sp500']['21']['n']))
V.raw('ag.n63', str(AG['sp500']['63']['n']))
V.put('acf.maxabs', max(AC[k]['max_abs_r'] for k in ['sp500', 'bet', 'btc']), 2)
V.put('acf.r2', 100 * max(AC[k]['max_abs_r'] for k in ['sp500', 'bet', 'btc']) ** 2, 1)
V.put('acf.sp500.ex', AC['sp500']['rho1_ex2020'], 2)
V.raw('nt.3.days.r', big(N['normal_tails']['3']['every_days']))


def mrow(k):
    return (f'{NAMES[k]} & @{{m.{k}.y0}} & @{{m.{k}.n}} & $@{{m.{k}.mean}}$ & $@{{m.{k}.sd}}$ & $@{{m.{k}.skew}}$ & '
            f'$@{{m.{k}.exkurt}}$ & $@{{m.{k}.min}}$')


def trow(k):
    return f'{NAMES[k]} & $@{{m.{k}.nu}}$ & @{{m.{k}.jb}} & $@{{m.{k}.daic}}$'


# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's Question and Route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: is the Normal distribution a good model for daily returns, and what do real returns look like?',
       '\\textbf{Întrebarea}: este distribuția Normală un model bun pentru randamentele zilnice și cum arată randamentele reale?'),
     [T('the answer decides how we measure risk, price options and test hypotheses',
        'de răspuns depind măsurarea riscului, evaluarea opțiunilor și testarea ipotezelor')]),
    (T('\\textbf{Route} of the chapter', '\\textbf{Traseul} capitolului'),
     [T('the Normal and the lognormal distributions, the central limit theorem (CLT)', 'distribuția Normală, distribuția lognormală și teorema limită centrală (CLT, central limit theorem)'),
      T('moments, the Jarque--Bera test and QQ plots', 'momente, testul Jarque--Bera și QQ plots'),
      T('the Student-$t$ distribution as a first heavy-tailed alternative', 'distribuția Student-$t$ ca primă alternativă cu cozi groase'),
      T('six stylised facts of returns \\refCont', 'șase fapte stilizate ale randamentelor \\refCont')]),
    T('Real data, @{y0}--@{y1}: S\\&P 500, DAX, BET, Bitcoin and four BVB stocks (SNP, BRD, SNN, SNG)',
      'Date reale, @{y0}--@{y1}: S\\&P 500, DAX, BET, Bitcoin și patru acțiuni BVB (SNP, BRD, SNN, SNG)')))

D.frame(T('Learning Outcomes', 'Rezultatele învățării'), items(
    T('Compute probabilities and quantiles of the Normal and lognormal distributions', 'Calculați probabilități și cuantile ale distribuțiilor Normală și lognormală'),
    T('State the CLT and say when the Normal approximation works', 'Enunțați CLT și precizați cînd funcționează aproximarea Normală'),
    T('Estimate skewness and excess kurtosis and test normality with the Jarque--Bera test', 'Estimați asimetria și excesul de boltire și testați normalitatea cu testul Jarque--Bera'),
    T('Read a QQ plot and fit a Student-$t$ distribution by maximum likelihood', 'Citiți un QQ plot și estimați o distribuție Student-$t$ prin metoda verosimilității maxime'),
    T('Check six stylised facts on a return series and explain what each means for a model', 'Verificați șase fapte stilizate pe o serie de randamente și explicați ce înseamnă fiecare pentru un model')))

D.frame(T('Reading and Tools', 'Bibliografie și instrumente'), items(
    (T('Textbook: \\refFHH, \\textit{Statistics of Financial Markets}, 5th ed., Sec.~3.3 (Normal and lognormal) and Sec.~11.2 (stylised facts)',
       'Manual: \\refFHH, \\textit{Statistics of Financial Markets}, ed. a 5-a, secț.~3.3 (Normală și lognormală) și secț.~11.2 (fapte stilizate)'),
     [T('exercises with solutions: \\refFHHex', 'exerciții rezolvate: \\refFHHex')]),
    T('Complementary: \\refTsay, Sec.~1.2 (distributional properties of returns); \\refCont', 'Complementar: \\refTsay, secț.~1.2 (proprietățile distribuției randamentelor); \\refCont'),
    (T('Python Quantlets of this chapter: \\href{https://github.com/danpele/SFM/tree/main/Quantlets/Ch_02}{Quantlets/Ch\\_02}',
       'Quantlet-urile Python ale capitolului: \\href{https://github.com/danpele/SFM/tree/main/Quantlets/Ch_02}{Quantlets/Ch\\_02}'),
     [T('ported from the SFE Quantlets of the textbook (SFEDaxReturnDistribution, SFElognormal, SFEclt, SFENormalApprox1--4)',
        'adaptate după Quantlet-urile SFE ale manualului (SFEDaxReturnDistribution, SFElognormal, SFEclt, SFENormalApprox1--4)'),
      T('each chart in these slides links to the Quantlet that draws it', 'fiecare grafic din aceste slide-uri are un link către Quantlet-ul care îl generează')]),
    T('Lecture notebook: \\href{\\colaburl{notebooks/EN/chapter2_lecture_notebook.ipynb}}{open in Google Colab}',
      'Notebook-ul cursului: \\href{\\colaburl{notebooks/EN/chapter2_lecture_notebook.ipynb}}{deschideți în Google Colab}'),
    T("Video course: \\quantinar{Tukey's g- and h- transformations}{https://quantinar.com/course/144/tukeys-g-and-h-transformations}",
      "Curs video: \\quantinar{Tukey's g- and h- transformations}{https://quantinar.com/course/144/tukeys-g-and-h-transformations}")))

# =============================================================================
# 1. DE CE CONTEAZĂ DISTRIBUȚIA
# =============================================================================
D.section('Why the Distribution Matters', 'Importanța distribuției')

D.frame(T('A Day That Should Not Happen', 'O zi care n-ar trebui să existe'), items(
    T('16 March 2020: the S\\&P 500 fell by $@{m.sp500.min}\\%$ (log return) in one day',
      '16 martie 2020: S\\&P 500 a scăzut cu $@{m.sp500.min}\\%$ (randament logaritmic) într-o singură zi'),
    (T('Daily standard deviation @{y0}--@{y1}: $@{m.sp500.sd}\\%$, so the fall was $@{m.sp500.zmin}$ standard deviations',
       'Abaterea standard zilnică @{y0}--@{y1}: $@{m.sp500.sd}\\%$, deci scăderea a fost de $@{m.sp500.zmin}$ abateri standard'),
     [T('under the Normal distribution, the probability of such a day is about $@{sp.pmin}$',
        'dacă randamentele ar urma distribuția Normală, probabilitatea unei asemenea zile ar fi de circa $@{sp.pmin}$'),
      T('that is, less than once in $10^{@{sp.pmin.exp}}$ years of trading', 'adică mai rar de o dată la $10^{@{sp.pmin.exp}}$ ani de tranzacționare')]),
    T('Yet such days happen: 19 October 1987, October 2008, March 2020, April 2025', 'Totuși, asemenea zile au loc: 19 octombrie 1987, octombrie 2008, martie 2020, aprilie 2025'),
    T('Conclusion: the model, not the market, is wrong', 'Concluzia: greșit este modelul, nu piața')))

D.frame(T('Question for the Room: How Often Is a 4-Sigma Day?', 'Întrebare pentru sală: cît de des apare o zi de 4 sigma?'), items(
    T('A \\textbf{$k$-sigma day}: a daily return more than $k$ standard deviations away from the mean',
      'O \\textbf{zi de $k$ sigma}: un randament zilnic aflat la mai mult de $k$ abateri standard de medie'),
    T('\\textbf{What do you think?} If daily returns were Normal, how often would a 4-sigma day occur?',
      '\\textbf{Ce credeți?} Dacă randamentele zilnice ar fi Normale, cît de des ar apărea o zi de 4 sigma?'),
    T('\\textbf{What do you think?} How often does it occur in the S\\&P 500 since @{y0}?',
      '\\textbf{Ce credeți?} Cît de des apare în S\\&P 500 din @{y0}?'),
    (T('\\textbf{Answer}', '\\textbf{Răspuns}'),
     [T('Normal distribution: once every @{sg.normal.every} trading days, about once in @{sg.normal.years} years',
        'distribuția Normală: o dată la @{sg.normal.every} zile de tranzacționare, cam o dată la @{sg.normal.years} de ani'),
      T('S\\&P 500: @{sg.sp500.4.obs} days in @{m.sp500.n} days, once every @{sg.sp500.every} days, about @{sg.sp500.peryear} per year',
        'S\\&P 500: @{sg.sp500.4.obs} zile din @{m.sp500.n}, o dată la @{sg.sp500.every} zile, cam @{sg.sp500.peryear} pe an')])))

D.frame(T('What a Distribution Is Used For', 'La ce folosește o distribuție'), items(
    (T('\\textbf{Risk}: the probability of a large loss', '\\textbf{Risc}: probabilitatea unei pierderi mari'),
     [T('VaR 1\\% (Value at Risk): the loss exceeded with probability 1\\%; a quantile of the distribution (Chapter 10)',
        'VaR 1\\% (Value at Risk, valoarea expusă la risc): pierderea depășită cu probabilitatea 1\\%; o cuantilă a distribuției (Capitolul 10)')]),
    (T('\\textbf{Pricing}: the Black--Scholes formula assumes lognormal prices', '\\textbf{Evaluare}: formula Black--Scholes presupune prețuri lognormale'),
     [T('heavier tails make deep out-of-the-money options more valuable', 'cozile mai groase fac mai valoroase opțiunile mult în afara banilor (deep out-of-the-money)')]),
    (T('\\textbf{Inference}: $t$-tests and confidence intervals assume (approximate) normality', '\\textbf{Inferență}: testele $t$ și intervalele de încredere presupun normalitate (aproximativă)'),
     [T('heavy tails make standard errors unreliable in small samples', 'cozile groase fac erorile standard nesigure în eșantioane mici')]),
    T('A wrong distribution gives wrong numbers, even with perfect data', 'O distribuție greșită produce rezultate greșite, chiar și cu date perfecte')))

D.frame(T('The Benchmark Model of This Chapter', 'Modelul de referință al capitolului'), items(
    (T('Price $P_t$, log return $r_t = \\ln(P_t/P_{t-1})$, so $P_t = P_{t-1}e^{r_t} = P_0 e^{r_1 + \\dots + r_t}$',
       'Prețul $P_t$, randamentul logaritmic $r_t = \\ln(P_t/P_{t-1})$, deci $P_t = P_{t-1}e^{r_t} = P_0 e^{r_1 + \\dots + r_t}$'),
     [T('notation: \\textbf{i.i.d.} = independent and identically distributed', 'notație: \\textbf{i.i.d.} = independente și identic distribuite')]),
    (T('\\textbf{Benchmark}: $r_t$ i.i.d. $N(\\mu, \\sigma^2)$', '\\textbf{Modelul de referință}: $r_t$ i.i.d. $N(\\mu, \\sigma^2)$'),
     [T('$N(\\mu, \\sigma^2)$: the Normal distribution with mean $\\mu$ and variance $\\sigma^2$ (daily mean and variance of the log returns)',
        '$N(\\mu, \\sigma^2)$: distribuția Normală cu media $\\mu$ și varianța $\\sigma^2$ (media și varianța zilnice ale randamentelor logaritmice)'),
      T('prices follow a geometric random walk; $P_t$ is lognormal \\refOsborne', 'prețurile urmează un mers aleator geometric; $P_t$ este lognormal \\refOsborne'),
      T('the first model of this kind, for arithmetic price changes: \\refBachelier', 'primul model de acest fel, pentru variațiile aritmetice ale prețurilor: \\refBachelier')]),
    T('This chapter tests each assumption of the benchmark in turn: normality, identical distribution and independence',
      'Capitolul verifică pe rînd ipotezele modelului de referință: normalitatea, distribuția identică și independența'),
    T('Textbook: \\refFHH, Sec.~3.3 and Sec.~11.2', 'Manual: \\refFHH, secț.~3.3 și secț.~11.2')))

# =============================================================================
# 2. DISTRIBUȚIA NORMALĂ
# =============================================================================
D.section('The Normal Distribution', 'Distribuția Normală')

D.frame(T('Carl Friedrich Gauss and the Normal Distribution', 'Carl Friedrich Gauss și distribuția Normală'), cols(items(
    T('De Moivre (1733) found the bell curve as the limit of binomial probabilities', 'De Moivre (1733) a obținut curba în formă de clopot ca limită a probabilităților binomiale'),
    T('Gauss (1809) used it for the errors of astronomical observations and derived least squares from it',
      'Gauss (1809) a folosit-o pentru erorile observațiilor astronomice și a dedus din ea metoda celor mai mici pătrate'),
    T('Laplace (1810) proved a first version of the central limit theorem', 'Laplace (1810) a demonstrat o primă versiune a teoremei limită centrale'),
    T('Hence its other name: the \\textbf{Gaussian} distribution', 'De aici și celălalt nume: distribuția \\textbf{gaussiană}'),
    T('In finance it became the default model of returns, because sums of Normal variables stay Normal',
      'În finanțe a devenit modelul implicit al randamentelor, pentru că sumele de variabile Normale rămîn Normale')),
    ph('gauss', 'Carl Friedrich Gauss (1777--1855)', h='0.56\\textheight'), wl='0.58', wr='0.38'))

D.frame(T('The Normal Distribution: Definition', 'Distribuția Normală: definiție'), items(
    (T('$X \\sim N(\\mu, \\sigma^2)$ has the \\textbf{PDF} (probability density function)', '$X \\sim N(\\mu, \\sigma^2)$ are \\textbf{PDF} (probability density function, densitatea de probabilitate)'),
     [T('$f(x) = \\dfrac{1}{\\sigma\\sqrt{2\\pi}} \\exp\\Big(-\\dfrac{(x - \\mu)^2}{2\\sigma^2}\\Big)$, \\quad $x \\in \\mathbb{R}$',
        '$f(x) = \\dfrac{1}{\\sigma\\sqrt{2\\pi}} \\exp\\Big(-\\dfrac{(x - \\mu)^2}{2\\sigma^2}\\Big)$, \\quad $x \\in \\mathbb{R}$'),
      T('two parameters: the mean $\\mu$ (location) and the variance $\\sigma^2$ (spread)', 'doi parametri: media $\\mu$ (poziția) și varianța $\\sigma^2$ (dispersia)')]),
    (T('\\textbf{Standard Normal distribution}: $Z = (X - \\mu)/\\sigma \\sim N(0, 1)$', '\\textbf{Distribuția Normală standard}: $Z = (X - \\mu)/\\sigma \\sim N(0, 1)$'),
     [T('its \\textbf{CDF} (cumulative distribution function) is $\\Phi(z) = P(Z \\le z)$, with no closed form; tables or software',
        '\\textbf{CDF} (cumulative distribution function, funcția de repartiție) este $\\Phi(z) = P(Z \\le z)$, fără formă închisă; se calculează cu tabele sau cu software')]),
    T('Probabilities of $X$ come from $Z$: $P(X \\le x) = \\Phi\\big((x - \\mu)/\\sigma\\big)$', 'Probabilitățile lui $X$ se obțin din $Z$: $P(X \\le x) = \\Phi\\big((x - \\mu)/\\sigma\\big)$'),
    (T('Quantile of order $\\alpha$: $x_\\alpha = \\mu + \\sigma z_\\alpha$, the value with $P(X \\le x_\\alpha) = \\alpha$', 'Cuantila de ordin $\\alpha$: $x_\\alpha = \\mu + \\sigma z_\\alpha$, valoarea pentru care $P(X \\le x_\\alpha) = \\alpha$'),
     [T('$z_\\alpha = \\Phi^{-1}(\\alpha)$: the quantile of $N(0, 1)$; $z_{0.01} = -2.33$, $z_{0.05} = -1.64$', '$z_\\alpha = \\Phi^{-1}(\\alpha)$: cuantila lui $N(0, 1)$; $z_{0.01} = -2.33$, $z_{0.05} = -1.64$')])))

D.frame(T('Properties Used in Finance', 'Proprietăți folosite în finanțe'), items(
    T('Symmetric around $\\mu$: mean = median = mode', 'Simetrică în jurul lui $\\mu$: media = mediana = modul'),
    T('Skewness $0$ and kurtosis $3$ (defined in Section 5)', 'Asimetria $0$ și boltirea $3$ (definite în secțiunea 5)'),
    (T('\\textbf{Closed under addition}: if $X \\sim N(\\mu_1, \\sigma_1^2)$ and $Y \\sim N(\\mu_2, \\sigma_2^2)$ are independent, then $X + Y \\sim N(\\mu_1 + \\mu_2, \\sigma_1^2 + \\sigma_2^2)$',
       '\\textbf{Închisă la adunare}: dacă $X \\sim N(\\mu_1, \\sigma_1^2)$ și $Y \\sim N(\\mu_2, \\sigma_2^2)$ sînt independente, atunci $X + Y \\sim N(\\mu_1 + \\mu_2, \\sigma_1^2 + \\sigma_2^2)$'),
     [T('$h$-day log return under the benchmark: $r_t(h) \\sim N(h\\mu, h\\sigma^2)$', 'randamentul logaritmic pe $h$ zile în modelul de referință: $r_t(h) \\sim N(h\\mu, h\\sigma^2)$'),
      T('volatility grows with $\\sqrt{h}$: the square-root-of-time rule used in Chapter 1', 'volatilitatea crește cu $\\sqrt{h}$: regula rădăcinii pătrate a timpului din Capitolul 1')]),
    T('Linear combinations of jointly Normal returns are Normal, so portfolio returns stay Normal', 'Orice combinație liniară de randamente cu distribuție Normală multivariată este Normală: randamentele portofoliilor rămîn Normale'),
    T('Fully described by two numbers: once $\\mu$ and $\\sigma$ are known, every probability is known', 'Complet determinată de doi parametri: dacă știm $\\mu$ și $\\sigma$, putem calcula orice probabilitate')))

chart(T('The 68--95--99.7 Rule', 'Regula 68--95--99,7'), 'sfm_ch2_normal_pdf', 'SFM_ch2_normal_lognormal', [
    T('Within 1, 2 and 3 standard deviations: @{nt.1.in}\\%, @{nt.2.in}\\% and @{nt.3.in}\\% of the probability',
      'La cel mult 1, 2 și 3 abateri standard de medie: @{nt.1.in}\\%, @{nt.2.in}\\% și @{nt.3.in}\\% din probabilitate'),
    T('Outside 3 standard deviations: one day in @{nt.3.days.r}; the tails of the Normal distribution fall off like $e^{-z^2/2}$, very fast',
      'Dincolo de 3 abateri standard: o zi din @{nt.3.days.r}; cozile distribuției Normale scad ca $e^{-z^2/2}$, foarte repede')], h='0.62\\textheight')

D.frame(T('How Rare Are Large Moves under the Normal Distribution?', 'Frecvența variațiilor mari în cazul distribuției Normale'), table(
    'crrr', '$k$ & $P(|Z| > k)$ & ' + T('once every \\dots\\ trading days', 'o dată la \\dots\\ zile de tranzacționare') + ' & ' + T('once every \\dots\\ years', 'o dată la \\dots\\ ani'),
    [f'{k} & ${{@{{nt.{k}.p}}}}$ & @{{nt.{k}.days}} & @{{nt.{k}.years}}' for k in range(1, 7)], size='footnotesize') + items(
    T('$P(|Z| > k)$: the probability that a standard Normal variable lies more than $k$ standard deviations from 0', '$P(|Z| > k)$: probabilitatea ca o variabilă Normală standard să se afle la mai mult de $k$ abateri standard de 0'),
    T('Years computed with 252 trading days a year', 'Anii sînt calculați cu 252 de zile de tranzacționare pe an'),
    T('Each extra standard deviation makes a move @{nt.rmin} to @{nt.rmax} times rarer', 'Fiecare abatere standard în plus face o variație de @{nt.rmin} pînă la @{nt.rmax} de ori mai rară'),
    T('A 6-sigma day should not happen in the whole history of stock markets', 'O zi de 6 sigma n-ar trebui să apară în toată istoria burselor')) + ql('SFM_ch2_normal_lognormal'))

D.frame(T('Worked Example: Large Losses of the S\\&P 500', 'Exemplu lucrat: pierderile mari ale S\\&P 500'), items(
    T('S\\&P 500, @{y0}--@{y1}: $n = @{m.sp500.n}$ daily log returns, mean $@{m.sp500.mean}\\%$, standard deviation $@{m.sp500.sd}\\%$',
      'S\\&P 500, @{y0}--@{y1}: $n = @{m.sp500.n}$ randamente logaritmice zilnice, media $@{m.sp500.mean}\\%$, abaterea standard $@{m.sp500.sd}\\%$'),
    (T('A loss larger than 3\\%: $z = (-3 - @{m.sp500.mean})/@{m.sp500.sd} = @{loss.sp500.3.z}$', 'O pierdere mai mare de 3\\%: $z = (-3 - @{m.sp500.mean})/@{m.sp500.sd} = @{loss.sp500.3.z}$'),
     [T('Normal distribution: $\\Phi(@{loss.sp500.3.z}) = @{sp.p3}\\%$, so $@{loss.sp500.3.exp}$ days expected', 'distribuția Normală: $\\Phi(@{loss.sp500.3.z}) = @{sp.p3}\\%$, deci numărul așteptat de zile este $@{loss.sp500.3.exp}$'),
      T('observed: @{loss.sp500.3.obs} days', 'numărul observat: @{loss.sp500.3.obs} zile')]),
    (T('A loss larger than 5\\%: $z = @{loss.sp500.5.z}$', 'O pierdere mai mare de 5\\%: $z = @{loss.sp500.5.z}$'),
     [T('Normal distribution: $@{loss.sp500.5.exp}$ days expected; observed: @{loss.sp500.5.obs} days', 'distribuția Normală: $@{loss.sp500.5.exp}$ zile așteptate; zile observate: @{loss.sp500.5.obs}')]),
    T('BET: @{loss.bet.5.obs} days below $-5\\%$ against $@{loss.bet.5.exp}$ expected', 'BET: @{loss.bet.5.obs} zile sub $-5\\%$, față de $@{loss.bet.5.exp}$ așteptate'),
    T('The Normal distribution underestimates the frequency of large losses by orders of magnitude', 'Distribuția Normală subestimează frecvența pierderilor mari cu cîteva ordine de mărime')), 'footnotesize')

D.recap(('The Normal Distribution', 'Distribuția Normală'), [
    T('Two parameters, symmetric, closed under addition, volatility scales with $\\sqrt{h}$', 'Doi parametri, simetrică, închisă la adunare, volatilitatea crește cu $\\sqrt{h}$'),
    T('Its tails are thin: a 4-sigma day once every @{sg.normal.years} years', 'Cozile ei sînt subțiri: o zi de 4 sigma o dată la @{sg.normal.years} de ani'),
    T('Real returns have many more large moves than it allows', 'Randamentele reale au mult mai multe variații mari decît permite ea')])

# =============================================================================
# 3. DISTRIBUȚIA LOGNORMALĂ
# =============================================================================
D.section('The Lognormal Distribution', 'Distribuția lognormală')

D.frame(T('From Normal Log Returns to Lognormal Prices', 'De la randamente logaritmice Normale la prețuri lognormale'), items(
    (T('A positive variable $Y$ is \\textbf{lognormal}, $Y \\sim LN(m, s^2)$, if $\\ln Y \\sim N(m, s^2)$', 'O variabilă pozitivă $Y$ este \\textbf{lognormală}, $Y \\sim LN(m, s^2)$, dacă $\\ln Y \\sim N(m, s^2)$'),
     [T('$m$, $s^2$: the mean and the variance of $\\ln Y$, not of $Y$', '$m$, $s^2$: media și varianța lui $\\ln Y$, nu ale lui $Y$')]),
    T('Density: $f(y) = \\dfrac{1}{y s\\sqrt{2\\pi}} \\exp\\Big(-\\dfrac{(\\ln y - m)^2}{2s^2}\\Big)$, \\quad $y > 0$',
      'Densitatea: $f(y) = \\dfrac{1}{y s\\sqrt{2\\pi}} \\exp\\Big(-\\dfrac{(\\ln y - m)^2}{2s^2}\\Big)$, \\quad $y > 0$'),
    (T('Under the benchmark, $\\ln(P_T/P_0) = r_1 + \\dots + r_T \\sim N(T\\mu, T\\sigma^2)$', 'În modelul de referință, $\\ln(P_T/P_0) = r_1 + \\dots + r_T \\sim N(T\\mu, T\\sigma^2)$'),
     [T('$T$: the horizon in days; so $P_T/P_0 \\sim LN(T\\mu, T\\sigma^2)$ \\refOsborne', '$T$: orizontul, în zile; deci $P_T/P_0 \\sim LN(T\\mu, T\\sigma^2)$ \\refOsborne')]),
    (T('Why lognormal and not Normal prices?', 'De ce prețuri lognormale și nu Normale?'),
     [T('prices cannot be negative; a Normal price could be', 'prețurile nu pot fi negative; un preț cu distribuție Normală ar putea fi negativ'),
      T('the simple return $R = e^r - 1 > -1$: you cannot lose more than you invested', 'randamentul simplu $R = e^r - 1 > -1$: nu puteți pierde mai mult decît ați investit')]),
    T('Textbook: \\refFHH, Sec.~3.3; the Black--Scholes model is built on it', 'Manual: \\refFHH, secț.~3.3; modelul Black--Scholes se bazează pe ea')))

D.frame(T('Mean, Median and Mode', 'Media, mediana și modul'), items(
    T('If $Y \\sim LN(m, s^2)$:', 'Dacă $Y \\sim LN(m, s^2)$:'),
    (T('median $e^{m}$; mode $e^{m - s^2}$; mean $E[Y] = e^{m + s^2/2}$', 'mediana $e^{m}$; modul $e^{m - s^2}$; media $E[Y] = e^{m + s^2/2}$'),
     [T('mode: the most likely value; median: the value exceeded with probability 1/2', 'modul: valoarea cea mai probabilă; mediana: valoarea depășită cu probabilitatea 1/2'),
      T('always mode $<$ median $<$ mean: a long right tail (positive skewness)', 'întotdeauna modul $<$ mediana $<$ media: o coadă dreaptă lungă (asimetrie pozitivă)')]),
    T('Variance: $\\text{Var}(Y) = (e^{s^2} - 1)\\,e^{2m + s^2}$', 'Varianța: $\\text{Var}(Y) = (e^{s^2} - 1)\\,e^{2m + s^2}$'),
    (T('The mean comes from $E[e^X] = e^{m + s^2/2}$ for $X \\sim N(m, s^2)$ (Seminar 2 derives it)',
       'Media rezultă din $E[e^X] = e^{m + s^2/2}$ pentru $X \\sim N(m, s^2)$ (Seminarul 2 o deduce)'),
     [T('the extra $s^2/2$ is the volatility drag of Chapter 1, seen from the other side', 'termenul suplimentar $s^2/2$ este volatility drag din Capitolul 1, privit din perspectiva prețurilor')]),
    T('Which one matters? The median is the outcome of a typical investor; the mean is pulled up by a few very good paths',
      'Care dintre ele contează? Mediana este rezultatul unui investitor tipic; media este ridicată de cîteva traiectorii foarte favorabile')))

chart(T('Lognormal Densities (SFElognormal)', 'Densități lognormale (SFElognormal)'), 'sfm_ch2_lognormal', 'SFM_ch2_normal_lognormal', [
    T('$m = 0$: the median is 1 for all three curves; the dashed lines mark the means', '$m = 0$: mediana este 1 pentru toate cele trei curbe; liniile punctate marchează mediile'),
    T('$s = 0.25$: mean $@{lnd.a.mean}$, mode $@{lnd.a.mode}$, almost symmetric; $s = 1$: mean $@{lnd.c.mean}$, mode $@{lnd.c.mode}$, skewness $@{lnd.c.skew}$',
      '$s = 0.25$: media $@{lnd.a.mean}$, modul $@{lnd.a.mode}$, aproape simetrică; $s = 1$: media $@{lnd.c.mean}$, modul $@{lnd.c.mode}$, asimetria $@{lnd.c.skew}$'),
    T('Long horizons or high volatility: the gap between mean and median widens', 'Orizonturi lungi sau volatilitate mare: distanța dintre medie și mediană crește')], h='0.62\\textheight')

D.frame(T('Worked Example: 100 Lei in the BET', 'Exemplu lucrat: 100 de lei în BET'), items(
    T('BET, @{y0}--@{y1}: annual mean log return $\\mu = @{ln.mu}\\%$ and volatility $\\sigma = @{ln.s}\\%$ (daily moments $\\times$ @{ln.ppy} and $\\sqrt{@{ln.ppy}}$)',
      'BET, @{y0}--@{y1}: randamentul logaritmic mediu anual $\\mu = @{ln.mu}\\%$ și volatilitatea $\\sigma = @{ln.s}\\%$ (momentele zilnice $\\times$ @{ln.ppy} și $\\sqrt{@{ln.ppy}}$)'),
    (T('After 1 year, $P_1/P_0 \\sim LN(\\mu, \\sigma^2)$', 'După un an, $P_1/P_0 \\sim LN(\\mu, \\sigma^2)$'),
     [T('median $100e^{\\mu} = @{ln.1.med}$; mean $100e^{\\mu + \\sigma^2/2} = @{ln.1.mean}$', 'mediana $100e^{\\mu} = @{ln.1.med}$; media $100e^{\\mu + \\sigma^2/2} = @{ln.1.mean}$'),
      T('probability of a loss $\\Phi(-\\mu/\\sigma) = @{ln.1.ploss}\\%$; 5\\% quantile $100e^{\\mu - 1.645\\sigma} = @{ln.1.q05}$',
        'probabilitatea unei pierderi $\\Phi(-\\mu/\\sigma) = @{ln.1.ploss}\\%$; cuantila de 5\\% $100e^{\\mu - 1.645\\sigma} = @{ln.1.q05}$'),
      T('$\\Phi(-\\mu/\\sigma) = P(r < 0)$: the probability of a negative log return; $-1.645 = z_{0.05}$', '$\\Phi(-\\mu/\\sigma) = P(r < 0)$: probabilitatea unui randament logaritmic negativ; $-1.645 = z_{0.05}$')]),
    (T('After 10 years, $P_{10}/P_0 \\sim LN(10\\mu, 10\\sigma^2)$', 'După 10 ani, $P_{10}/P_0 \\sim LN(10\\mu, 10\\sigma^2)$'),
     [T('median @{ln.10.med}; mean @{ln.10.mean}; probability of a loss @{ln.10.ploss}\\%; 5\\% quantile @{ln.10.q05}',
        'mediana @{ln.10.med}; media @{ln.10.mean}; probabilitatea unei pierderi @{ln.10.ploss}\\%; cuantila de 5\\% @{ln.10.q05}')]),
    (T('The probability of a loss falls with the horizon; after 10 years even the 5\\% quantile is above 100, but only if the historical mean persists', 'Probabilitatea unei pierderi scade cu orizontul; după 10 ani, chiar și cuantila de 5\\% depășește 100, dar numai dacă media istorică se menține') if Q05_10 > 100 else T('The probability of a loss falls with the horizon, but the 5\\% quantile stays below 100 even after 10 years', 'Probabilitatea unei pierderi scade cu orizontul, dar cuantila de 5\\% rămîne sub 100 chiar și după 10 ani')),
    T('Caveat: these numbers inherit every weakness of the Normal model of log returns', 'Atenție: aceste rezultate moștenesc toate slăbiciunile modelului Normal al randamentelor logaritmice')), 'footnotesize')

D.recap(('The Lognormal Distribution', 'Distribuția lognormală'), [
    T('Normal log returns $\\Rightarrow$ lognormal prices: positive, right-skewed', 'Randamente logaritmice Normale $\\Rightarrow$ prețuri lognormale: pozitive, asimetrice la dreapta'),
    T('Median $e^m$ below mean $e^{m + s^2/2}$: the gap is the volatility drag', 'Mediana $e^m$ sub media $e^{m + s^2/2}$: diferența este volatility drag'),
    T('Report the median and a low quantile, not only the mean', 'Raportați mediana și o cuantilă joasă, nu doar media')])

# =============================================================================
# 4. TEOREMA LIMITĂ CENTRALĂ
# =============================================================================
D.section('The Central Limit Theorem', 'Teorema limită centrală')

D.frame(T('The Galton Board', 'Tabla lui Galton'), cols(items(
    T('Francis Galton (1889): balls fall through rows of pins and bounce left or right with probability $1/2$',
      'Francis Galton (1889): bilele cad printre rînduri de cuie și sar la stînga sau la dreapta cu probabilitatea $1/2$'),
    T('The final position of a ball is a \\textbf{sum} of many small independent shocks', 'Poziția finală a unei bile este o \\textbf{sumă} de multe șocuri mici independente'),
    T('The heaps form a bell curve: binomial probabilities close to the Normal density', 'Bilele acumulate formează o curbă în formă de clopot: probabilități binomiale apropiate de densitatea Normală'),
    T('The same argument was used for returns: a daily return is the sum of many small price changes during the day',
      'Același argument a fost folosit pentru randamente: un randament zilnic este suma multor variații mici de preț din timpul zilei')),
    ph('galton', T('A Galton board after the balls have fallen', 'O tablă Galton după căderea bilelor'), h='0.45\\textheight'), wl='0.52', wr='0.44'))

D.frame(T('The Central Limit Theorem', 'Teorema limită centrală'), items(
    (T('\\textbf{CLT} (Lindeberg--Lévy): $X_1, \\dots, X_n$ i.i.d. with mean $\\mu$ and finite variance $\\sigma^2$',
       '\\textbf{CLT} (Lindeberg--Lévy): $X_1, \\dots, X_n$ i.i.d. cu media $\\mu$ și varianța finită $\\sigma^2$'),
     [T('$\\dfrac{\\sqrt{n}\\,(\\bar X_n - \\mu)}{\\sigma} \\xrightarrow{d} N(0, 1)$ as $n \\to \\infty$', '$\\dfrac{\\sqrt{n}\\,(\\bar X_n - \\mu)}{\\sigma} \\xrightarrow{d} N(0, 1)$ cînd $n \\to \\infty$'),
      T('$\\bar X_n$: the mean of the $n$ variables; $\\xrightarrow{d}$: convergence in distribution, the CDF of the left side approaches $\\Phi$',
        '$\\bar X_n$: media celor $n$ variabile; $\\xrightarrow{d}$: convergență în distribuție, funcția de repartiție a membrului stîng se apropie de $\\Phi$'),
      T('equivalently, the sum $X_1 + \\dots + X_n$ is approximately $N(n\\mu, n\\sigma^2)$', 'echivalent, suma $X_1 + \\dots + X_n$ este aproximativ $N(n\\mu, n\\sigma^2)$')]),
    (T('Three conditions', 'Trei condiții'),
     [T('independence; identical distribution; finite variance', 'independență; distribuție identică; varianță finită')]),
    T('How fast? The error falls like $1/\\sqrt{n}$ and grows with the skewness of $X$ (Berry--Esseen bound)',
      'Cît de repede? Eroarea scade ca $1/\\sqrt{n}$ și crește cu asimetria lui $X$ (marginea Berry--Esseen)'),
    T('It explains why the Normal distribution is everywhere, and why sample means have Normal confidence intervals',
      'Explică de ce distribuția Normală apare peste tot și de ce intervalele de încredere pentru medie se construiesc cu distribuția Normală')))

chart(T('The CLT at Work (SFEclt)', 'CLT în practică (SFEclt)'), 'sfm_ch2_clt', 'SFM_ch2_clt_normal_approx', [
    T('Standardised mean of $n$ Bernoulli($0.2$) draws (a skewed variable), 20\\,000 simulations', 'Media standardizată a $n$ extrageri Bernoulli($0.2$) (o variabilă asimetrică), 20\\,000 de simulări'),
    T('Share above $1.96$: @{clt.5.tail}\\% for $n = 5$, @{clt.30.tail}\\% for $n = 30$, @{clt.300.tail}\\% for $n = 300$; Normal distribution: @{clt.ntail}\\%',
      'Ponderea peste $1.96$: @{clt.5.tail}\\% pentru $n = 5$, @{clt.30.tail}\\% pentru $n = 30$, @{clt.300.tail}\\% pentru $n = 300$; distribuția Normală: @{clt.ntail}\\%'),
    T('Skewness of the mean, $(1 - 2p)/\\sqrt{np(1-p)}$ with $p = 0.2$: $@{clt.5.skew}$, $@{clt.30.skew}$, $@{clt.300.skew}$; it falls like $1/\\sqrt{n}$', 'Asimetria mediei, $(1 - 2p)/\\sqrt{np(1-p)}$ cu $p = 0.2$: $@{clt.5.skew}$, $@{clt.30.skew}$, $@{clt.300.skew}$; scade ca $1/\\sqrt{n}$')],
      h='0.60\\textheight')

chart(T('The Normal Approximation of the Binomial (SFENormalApprox)', 'Aproximarea Normală a distribuției binomiale (SFENormalApprox)'), 'sfm_ch2_normal_approx',
      'SFM_ch2_clt_normal_approx', [
          T('$B(n, p)$: the number of successes in $n$ independent trials with success probability $p$; $B(n, p) \\approx N\\big(np, np(1-p)\\big)$',
            '$B(n, p)$: numărul de succese în $n$ încercări independente, cu probabilitatea de succes $p$; $B(n, p) \\approx N\\big(np, np(1-p)\\big)$'),
          T('Continuity correction: $P(X \\le k) \\approx \\Phi\\big((k + 0.5 - np)/\\sqrt{np(1-p)}\\big)$; the $+0.5$ spreads each integer $k$ over $[k - 0.5, k + 0.5]$',
            'Corecția de continuitate: $P(X \\le k) \\approx \\Phi\\big((k + 0.5 - np)/\\sqrt{np(1-p)}\\big)$; termenul $+0.5$ distribuie fiecare întreg $k$ pe intervalul $[k - 0.5; k + 0.5]$'),
          T('Largest error of the CDF: $@{ap.20_0.5.err}$ for $n = 20, p = 0.5$; $@{ap.20_0.1.err}$ for $n = 20, p = 0.1$; $@{ap.100_0.1.err}$ for $n = 100, p = 0.1$',
            'Cea mai mare eroare a CDF: $@{ap.20_0.5.err}$ pentru $n = 20, p = 0.5$; $@{ap.20_0.1.err}$ pentru $n = 20, p = 0.1$; $@{ap.100_0.1.err}$ pentru $n = 100, p = 0.1$'),
          T('Rule of thumb: the approximation is good when $np(1-p)$ is large (at least about 9)', 'Regulă practică: aproximarea este bună cînd $np(1-p)$ este mare (cel puțin circa 9)')],
      h='0.54\\textheight')

D.frame(T('Question for the Room: Does the CLT Make Daily Returns Normal?', 'Întrebare pentru sală: devin randamentele zilnice Normale datorită CLT?'), items(
    T('A daily log return is the sum of thousands of intraday log returns', 'Un randament logaritmic zilnic este suma a mii de randamente logaritmice din timpul zilei'),
    T('\\textbf{What do you think?} Should daily returns then be close to the Normal distribution?', '\\textbf{Ce credeți?} Ar trebui atunci randamentele zilnice să fie aproape Normale?'),
    (T('\\textbf{Answer}: not necessarily; each condition of the CLT can fail', '\\textbf{Răspuns}: nu neapărat; fiecare condiție a CLT poate fi încălcată'),
     [T('dependence: calm and turbulent days come in clusters (Section 8)', 'dependență: zilele calme și cele agitate apar grupat (secțiunea 8)'),
      T('changing distribution: volatility differs from day to day', 'distribuție schimbătoare: volatilitatea diferă de la o zi la alta'),
      T('infinite variance: \\refMandelbrot{} proposed it for cotton prices (Chapter 3, $\\alpha$-stable distributions)',
        'varianță infinită: \\refMandelbrot{} a propus-o pentru prețurile bumbacului (Capitolul 3, distribuții $\\alpha$-stabile)')]),
    T('The data decide: the next sections measure how far returns are from the Normal distribution', 'Datele decid: secțiunile următoare măsoară cît de departe sînt randamentele de distribuția Normală')))

D.recap(('The Central Limit Theorem', 'Teorema limită centrală'), [
    T('Sums of many i.i.d. terms with finite variance are approximately Normal', 'Sumele multor termeni i.i.d. cu varianță finită sînt aproximativ Normale'),
    T('Convergence is slower for skewed variables; $np(1-p) \\ge 9$ for the binomial', 'Convergența este mai lentă pentru variabile asimetrice; $np(1-p) \\ge 9$ pentru distribuția binomială'),
    T('Returns may violate independence, identical distribution or finite variance', 'Randamentele pot încălca independența, distribuția identică sau varianța finită')])

# =============================================================================
# 5. MOMENTE ȘI FORMĂ
# =============================================================================
D.section('Moments and the Shape of a Distribution', 'Momentele și forma unei distribuții')

D.frame(T('Four Moments', 'Patru momente'), items(
    T('Mean $\\mu = E[X]$; variance $\\sigma^2 = E[(X - \\mu)^2]$; $E[\\cdot]$ is the expected value', 'Media $\\mu = E[X]$; varianța $\\sigma^2 = E[(X - \\mu)^2]$; $E[\\cdot]$ este valoarea așteptată'),
    (T('\\textbf{Skewness}: $\\gamma_1 = E[(X - \\mu)^3]/\\sigma^3$', '\\textbf{Asimetria} (skewness): $\\gamma_1 = E[(X - \\mu)^3]/\\sigma^3$'),
     [T('the third power keeps the sign of each deviation; $\\gamma_1 < 0$: a longer left tail, large losses more frequent than large gains', 'puterea a treia păstrează semnul fiecărei abateri; $\\gamma_1 < 0$: o coadă stîngă mai lungă, pierderile mari mai frecvente decît cîștigurile mari')]),
    (T('\\textbf{Kurtosis}: $\\gamma_2 = E[(X - \\mu)^4]/\\sigma^4$; Normal distribution: $\\gamma_2 = 3$', '\\textbf{Boltirea} (kurtosis): $\\gamma_2 = E[(X - \\mu)^4]/\\sigma^4$; distribuția Normală: $\\gamma_2 = 3$'),
     [T('\\textbf{excess kurtosis} $\\gamma_2 - 3$: positive means heavy tails and a tall, narrow centre (\\textbf{leptokurtic})',
        '\\textbf{excesul de boltire} $\\gamma_2 - 3$: pozitiv înseamnă cozi groase și un centru înalt și îngust (distribuție \\textbf{leptocurtică})')]),
    T('Both are free of units: they describe the shape, not the location or the scale', 'Ambele sînt adimensionale: descriu forma, nu poziția sau scala'),
    T('Fourth powers make the kurtosis very sensitive to a few extreme days', 'Puterile a patra fac boltirea foarte sensibilă la cîteva zile extreme')))

chart(T('What Skewness and Kurtosis Look Like', 'Asimetria și boltirea în grafice'), 'sfm_ch2_shapes', 'SFM_ch2_moments_jarque_bera', [
    T('Left: same mean and variance; the Student-$t(5)$ curve (Section 7) has excess kurtosis 6: more mass in the centre and in the tails, less in the shoulders',
      'Stînga: aceeași medie și varianță; curba Student-$t(5)$ (secțiunea 7) are exces de boltire 6: mai multă masă în centru și în cozi, mai puțină în umeri'),
    T('Right: skew-Normal densities with skewness $\\pm 0.85$; the dashed curve is the Normal distribution', 'Dreapta: densități skew-Normal cu asimetria $\\pm 0.85$; curba punctată este distribuția Normală')], h='0.62\\textheight')

D.frame(T('Sample Moments', 'Momentele de selecție'), items(
    T('Central sample moments: $m_k = \\frac1n \\sum_{t=1}^n (r_t - \\bar r)^k$, the average $k$-th power of the deviations from the mean ($m_2$: variance)', 'Momentele centrate de selecție: $m_k = \\frac1n \\sum_{t=1}^n (r_t - \\bar r)^k$, media puterii $k$ a abaterilor de la medie ($m_2$: varianța)'),
    T('Sample skewness $S = m_3/m_2^{3/2}$; sample excess kurtosis $K = m_4/m_2^2 - 3$', 'Asimetria de selecție $S = m_3/m_2^{3/2}$; excesul de boltire de selecție $K = m_4/m_2^2 - 3$'),
    (T('If the data are i.i.d. Normal: $S \\approx N(0, 6/n)$ and $K \\approx N(0, 24/n)$', 'Dacă datele sînt i.i.d. Normale: $S \\approx N(0, 6/n)$ și $K \\approx N(0, 24/n)$'),
     [T('with $n = @{m.sp500.n}$: standard errors $\\sqrt{6/n} = @{m.sp500.sesk}$ and $\\sqrt{24/n} = @{m.sp500.seku}$', 'cu $n = @{m.sp500.n}$: erorile standard $\\sqrt{6/n} = @{m.sp500.sesk}$ și $\\sqrt{24/n} = @{m.sp500.seku}$')]),
    T('These standard errors hold only under normality; for heavy-tailed data they are far too small (Seminar 2, B1)',
      'Aceste erori standard sînt valabile doar în ipoteza de normalitate; pentru date cu cozi groase sînt mult prea mici (Seminarul 2, B1)'),
    T('Textbook exercises: \\refFHHex', 'Exerciții din manual: \\refFHHex')))

D.frame(T('Daily Log Returns: Moments, @{y0}--@{y1}', 'Randamente logaritmice zilnice: momente, @{y0}--@{y1}'),
        table('lrrrrrrr', T('Series', 'Seria') + ' & ' + T('from', 'din') + ' & $n$ & ' + T('Mean \\%', 'Media \\%') + ' & ' + T('S.d. \\%', 'Ab. std. \\%') + ' & '
              + T('Skew.', 'Asim.') + ' & ' + T('Exc. kurt.', 'Exces bolt.') + ' & Min \\%', [mrow(k) for k in ASSETS], size='footnotesize') + items(
            T('Every series has excess kurtosis between @{ku.min} (@{ku.min.name}) and @{ku.max} (@{ku.max.name}): far from 0', 'Toate seriile au exces de boltire între @{ku.min} (@{ku.min.name}) și @{ku.max} (@{ku.max.name}): departe de 0'),
            T('Every series except SNN has negative skewness', 'Toate seriile, cu excepția SNN, au asimetrie negativă'),
            T('Worst days: S\\&P 500 on @{m.sp500.mindate}, BET on @{m.bet.mindate} (OUG 114, a tax on bank assets), Bitcoin on @{m.btc.mindate}',
              'Cele mai slabe zile: S\\&P 500 pe @{m.sp500.mindate}, BET pe @{m.bet.mindate} (OUG 114, taxa pe activele bancare), Bitcoin pe @{m.btc.mindate}')) + ql('SFM_ch2_moments_jarque_bera'),
        'footnotesize')

chart(T('DAX Returns vs the Normal Density (SFEDaxReturnDistribution)', 'Randamentele DAX față de densitatea Normală (SFEDaxReturnDistribution)'), 'sfm_ch2_dax_density',
      'SFM_ch2_moments_jarque_bera', [
          T('Kernel density estimate (a smoothed histogram) against the Normal density with the same mean and standard deviation',
            'Estimatorul nucleu al densității (o histogramă netezită) față de densitatea Normală cu aceeași medie și abatere standard'),
          T('Centre: peak @{dax.peak} vs @{dax.npeak}; @{dax.in1}\\% of days within one standard deviation, vs @{dax.nin1}\\% for the Normal distribution',
            'Centrul: vîrful densității @{dax.peak} față de @{dax.npeak}; @{dax.in1}\\% din zile în interiorul unei abateri standard, față de @{dax.nin1}\\% pentru distribuția Normală'),
          T('Tails (log scale): at $-6\\%$ the estimated density is @{dax.ratio6} times the Normal one', 'Cozile (scară logaritmică): la $-6\\%$ densitatea estimată este de @{dax.ratio6} de ori mai mare decît densitatea Normală')],
      h='0.60\\textheight')

D.frame(T('Check the Data Before Measuring Tails', 'Verificați datele înainte de a măsura cozile'), items(
    T('Banca Transilvania (TLV), 27--31 May 2016: the close falls from @{tlv.c0} to @{tlv.c1} lei, the ex-date of a bonus share issue',
      'Banca Transilvania (TLV), 27--31 mai 2016: prețul de închidere scade de la @{tlv.c0} la @{tlv.c1} lei, data ex a unei distribuiri de acțiuni gratuite'),
    (T('The adjusted close in our data: @{tlv.a0}, then @{tlv.a1}, then @{tlv.a2}', 'Prețul ajustat din datele noastre: @{tlv.a0}, apoi @{tlv.a1}, apoi @{tlv.a2}'),
     [T('log returns $@{tlv.r1}\\%$ and $+@{tlv.r2}\\%$ on two consecutive days: the adjustment was applied one day late',
        'randamente logaritmice $@{tlv.r1}\\%$ și $+@{tlv.r2}\\%$ în două zile consecutive: ajustarea a fost aplicată cu o zi mai tîrziu')]),
    T('Excess kurtosis of TLV, @{y0}--@{y1}: @{tlv.k} with these two days, @{tlv.kc} without them', 'Excesul de boltire al TLV, @{y0}--@{y1}: @{tlv.k} cu aceste două zile, @{tlv.kc} fără ele'),
    T('Two wrong days out of thousands change the kurtosis by a third: this is why TLV is not in the tables of this chapter',
      'Două zile greșite din mii schimbă boltirea cu o treime: de aceea TLV nu apare în tabelele acestui capitol'),
    T('Rule from Chapter 1: list the largest moves and check each one, especially pairs of opposite jumps',
      'Regula din Capitolul 1: listați cele mai mari variații și verificați-le pe fiecare, mai ales perechile de salturi de semn opus')), 'footnotesize')

D.recap(('Moments', 'Momente'), [
    T('Skewness measures asymmetry, excess kurtosis measures tail weight relative to the Normal distribution', 'Asimetria măsoară lipsa simetriei, excesul de boltire măsoară greutatea cozilor față de distribuția Normală'),
    T('Daily returns: mostly negative skewness and excess kurtosis between @{ku.min0} and @{ku.max0}', 'Randamentele zilnice: în general asimetrie negativă și exces de boltire între @{ku.min0} și @{ku.max0}'),
    T('Kurtosis is driven by a few days: check them before trusting it', 'Boltirea este determinată de cîteva zile: verificați-le înainte de a avea încredere în ea')])

# =============================================================================
# 6. TESTAREA NORMALITĂȚII
# =============================================================================
D.section('Testing Normality: Jarque--Bera and QQ Plots', 'Testarea normalității: Jarque--Bera și QQ plots')

D.frame(T('The Jarque--Bera Test', 'Testul Jarque--Bera'), items(
    T('$H_0$: the data are Normal, so skewness $= 0$ and excess kurtosis $= 0$', '$H_0$: datele sînt Normale, deci asimetria $= 0$ și excesul de boltire $= 0$'),
    (T('\\textbf{JB} statistic \\refJBa, \\refJBb: $\\text{JB} = \\dfrac{n}{6}\\Big(S^2 + \\dfrac{K^2}{4}\\Big)$', 'Statistica \\textbf{JB} \\refJBa, \\refJBb: $\\text{JB} = \\dfrac{n}{6}\\Big(S^2 + \\dfrac{K^2}{4}\\Big)$'),
     [T('the sum of two squared standardised statistics: $(S/\\sqrt{6/n})^2 + (K/\\sqrt{24/n})^2$', 'suma a două statistici standardizate la pătrat: $(S/\\sqrt{6/n})^2 + (K/\\sqrt{24/n})^2$')]),
    (T('Under $H_0$ and for large $n$: $\\text{JB} \\sim \\chi^2(2)$; reject at 5\\% if $\\text{JB} > @{jb.crit}$', 'În ipoteza $H_0$ și pentru $n$ mare: $\\text{JB} \\sim \\chi^2(2)$; respingem la 5\\% dacă $\\text{JB} > @{jb.crit}$'),
     [T('$\\chi^2(2)$: the chi-squared distribution with 2 degrees of freedom, the distribution of a sum of two squared independent $N(0,1)$ variables; @{jb.crit} is its 95\\% quantile',
        '$\\chi^2(2)$: distribuția hi-pătrat cu 2 grade de libertate, distribuția sumei pătratelor a două variabile $N(0,1)$ independente; @{jb.crit} este cuantila ei de 95\\%'),
      T('JB $= 0$ for a perfect Normal shape; large values mean skewness or excess kurtosis far from 0', 'JB $= 0$ pentru o formă perfect Normală; valorile mari indică asimetrie sau exces de boltire departe de 0')]),
    T('Simple, uses only two moments, and is the standard test reported in empirical finance', 'Este simplu, folosește doar două momente și este testul raportat de regulă în finanțele empirice')))

D.frame(T('Worked Example: Jarque--Bera for the S\\&P 500', 'Exemplu lucrat: Jarque--Bera pentru S\\&P 500'), items(
    T('$n = @{m.sp500.n}$, $S = @{m.sp500.skew}$, $K = @{m.sp500.exkurt}$', '$n = @{m.sp500.n}$, $S = @{m.sp500.skew}$, $K = @{m.sp500.exkurt}$'),
    T('$\\text{JB} = @{jb.n6} \\times (@{jb.s2} + @{jb.k24}) = @{m.sp500.jb}$', '$\\text{JB} = @{jb.n6} \\times (@{jb.s2} + @{jb.k24}) = @{m.sp500.jb}$'),
    T('Critical value @{jb.crit}: normality is rejected; the p-value is zero to any number of decimals', 'Valoarea critică @{jb.crit}: normalitatea este respinsă; p-value-ul este practic zero'),
    T('The kurtosis term dominates: almost all of JB comes from $K^2/4$', 'Termenul de boltire domină: aproape toată valoarea JB provine din $K^2/4$'),
    (T('JB for the other series: BET @{m.bet.jb}, DAX @{m.dax.jb}, Bitcoin @{m.btc.jb}, SNN @{m.snn.jb}', 'JB pentru celelalte serii: BET @{m.bet.jb}, DAX @{m.dax.jb}, Bitcoin @{m.btc.jb}, SNN @{m.snn.jb}'),
     [T('all rejected at any usual level', 'toate respinse la orice nivel uzual')])))

D.frame(T('What Jarque--Bera Does Not Tell You', 'Limitele testului Jarque--Bera'), items(
    (T('With thousands of observations it rejects even tiny departures', 'Cu mii de observații respinge chiar și abateri foarte mici'),
     [T('report the size of $S$ and $K$, not only the p-value', 'raportați mărimea lui $S$ și $K$, nu doar p-value-ul')]),
    (T('It assumes i.i.d. data; with volatility clustering its standard errors are too small', 'Testul presupune date i.i.d.; cu volatility clustering, erorile standard ale lui $S$ și $K$ sînt prea mici'),
     [T('S\\&P 500: Normal-theory 95\\% interval for $K$: $[@{bs.nlo}, @{bs.nhi}]$; bootstrap interval: $[@{bs.klo}, @{bs.khi}]$ (Seminar 2, B1)',
        'S\\&P 500: intervalul de 95\\% pentru $K$ în ipoteza de normalitate: $[@{bs.nlo}, @{bs.nhi}]$; intervalul bootstrap: $[@{bs.klo}, @{bs.khi}]$ (Seminarul 2, B1)')]),
    T('A rejection says the data are not Normal; it does not say which distribution they follow', 'Respingerea arată că datele nu sînt Normale, dar nu indică ce distribuție urmează'),
    T('Other tests (Kolmogorov--Smirnov, Anderson--Darling) compare the whole CDF: Chapter 6', 'Alte teste (Kolmogorov--Smirnov, Anderson--Darling) compară întreaga CDF: Capitolul 6')))

D.frame(T('QQ Plots', 'QQ plots'), items(
    (T('A \\textbf{QQ plot} (quantile--quantile plot) compares the empirical quantiles with those of a model \\refWG',
       'Un \\textbf{QQ plot} (quantile--quantile plot, graficul cuantilă--cuantilă) compară cuantilele empirice cu cele ale unui model \\refWG'),
     [T('sort the data: $r_{(1)} \\le r_{(2)} \\le \\dots \\le r_{(n)}$', 'ordonați datele: $r_{(1)} \\le r_{(2)} \\le \\dots \\le r_{(n)}$'),
      T('plot $r_{(i)}$ against the model quantile $F^{-1}\\big((i - 0.5)/n\\big)$', 'reprezentați $r_{(i)}$ în funcție de cuantila modelului $F^{-1}\\big((i - 0.5)/n\\big)$'),
      T('$r_{(i)}$: the $i$-th smallest return; $F^{-1}$: the quantile function of the model; $(i - 0.5)/n$: the share of the data below $r_{(i)}$',
        '$r_{(i)}$: al $i$-lea cel mai mic randament; $F^{-1}$: funcția cuantilă a modelului; $(i - 0.5)/n$: proporția datelor aflate sub $r_{(i)}$')]),
    (T('Reading', 'Interpretare'),
     [T('points on the 45-degree line: the model fits', 'puncte pe prima bisectoare: modelul se potrivește'),
      T('an S-shape, low end below and high end above the line: heavier tails than the model', 'o formă de S, capătul stîng sub dreaptă și cel drept deasupra: cozi mai groase decît ale modelului'),
      T('only one end bent away: skewness', 'un singur capăt se îndepărtează de dreaptă: asimetrie')]),
    T('Unlike JB, it shows \\textbf{where} the model fails: centre, shoulders or tails', 'Spre deosebire de JB, arată \\textbf{unde} greșește modelul: în centru, în umeri sau în cozi')))

chart(T('QQ Plots against the Normal Distribution', 'QQ plots față de distribuția Normală'), 'sfm_ch2_qq_normal', 'SFM_ch2_qq_plots', [
    T('Standardised returns against $N(0,1)$ quantiles: all three bend away from the line at both ends', 'Randamente standardizate față de cuantilele $N(0,1)$: toate trei se depărtează de dreaptă la ambele capete'),
    T('S\\&P 500: lowest standardised return $@{qq.sp500.min}$ where the Normal quantile is $@{qq.sp500.tmin}$; BET: $@{qq.bet.min}$ vs $@{qq.bet.tmin}$',
      'S\\&P 500: cel mai mic randament standardizat $@{qq.sp500.min}$, unde cuantila Normală este $@{qq.sp500.tmin}$; BET: $@{qq.bet.min}$ față de $@{qq.bet.tmin}$'),
    T('The lower tail bends more than the upper one: negative skewness', 'Coada stîngă se curbează mai mult decît cea dreaptă: asimetrie negativă')], h='0.62\\textheight')

D.recap(('Testing Normality', 'Testarea normalității'), [
    T('JB combines skewness and kurtosis; it rejects normality for every series', 'JB combină asimetria și boltirea; respinge normalitatea pentru toate seriile'),
    T('Report the size of the departure; Normal-theory standard errors are too small', 'Raportați mărimea abaterii; erorile standard din teoria Normală sînt prea mici'),
    T('QQ plots show where the model fails: here, in both tails', 'QQ plots arată unde greșește modelul: aici, în ambele cozi')])

# =============================================================================
# 7. STUDENT-t
# =============================================================================
D.section('The Student-$t$ Distribution', 'Distribuția Student-$t$')

D.frame(T('"Student": William Sealy Gosset', '„Student”: William Sealy Gosset'), cols(items(
    T('Chemist and brewer at Guinness, Dublin, working with small samples of barley and hops', 'Chimist și berar la Guinness, Dublin, lucra cu eșantioane mici de orz și hamei'),
    T('The brewery did not allow its staff to publish under their own names: he signed ``Student\'\' \\refStudent',
      'Fabrica de bere nu își lăsa angajații să publice sub propriul nume: a semnat „Student” \\refStudent'),
    T('He derived the distribution of $(\\bar X - \\mu)/(s/\\sqrt n)$ for Normal data: the $t$ distribution', 'A dedus distribuția lui $(\\bar X - \\mu)/(s/\\sqrt n)$ pentru date Normale: distribuția $t$'),
    T('In finance it is used for a different reason: its tails are heavier than those of the Normal distribution \\refPraetz, \\refBG',
      'În finanțe este folosită din alt motiv: cozile ei sînt mai groase decît cele ale distribuției Normale \\refPraetz, \\refBG')),
    ph('gosset', 'William Sealy Gosset (1876--1937)', h='0.56\\textheight'), wl='0.58', wr='0.38'))

D.frame(T('The Student-$t$ Distribution: Definition', 'Distribuția Student-$t$: definiție'), items(
    (T('$T = Z/\\sqrt{V/\\nu}$ with $Z \\sim N(0,1)$ independent of $V \\sim \\chi^2(\\nu)$: $T \\sim t(\\nu)$', '$T = Z/\\sqrt{V/\\nu}$, cu $Z \\sim N(0,1)$ independentă de $V \\sim \\chi^2(\\nu)$: $T \\sim t(\\nu)$'),
     [T('$\\nu > 0$: the \\textbf{degrees of freedom}, which control the tails', '$\\nu > 0$: \\textbf{gradele de libertate}, care controlează cozile'),
      T('$\\chi^2(\\nu)$: the distribution of a sum of $\\nu$ squared independent $N(0,1)$ variables; $\\Gamma$: the gamma function, $\\Gamma(k) = (k-1)!$ for integer $k$',
        '$\\chi^2(\\nu)$: distribuția sumei pătratelor a $\\nu$ variabile $N(0,1)$ independente; $\\Gamma$: funcția gamma, $\\Gamma(k) = (k-1)!$ pentru $k$ întreg')]),
    T('Density: $f(x) = \\dfrac{\\Gamma\\big((\\nu+1)/2\\big)}{\\sqrt{\\nu\\pi}\\,\\Gamma(\\nu/2)} \\Big(1 + \\dfrac{x^2}{\\nu}\\Big)^{-(\\nu+1)/2}$',
      'Densitatea: $f(x) = \\dfrac{\\Gamma\\big((\\nu+1)/2\\big)}{\\sqrt{\\nu\\pi}\\,\\Gamma(\\nu/2)} \\Big(1 + \\dfrac{x^2}{\\nu}\\Big)^{-(\\nu+1)/2}$'),
    T('Location--scale version for returns: $r = m + s\\,T$, with location $m$, scale $s$ and $\\nu$', 'Versiunea cu parametri de poziție și de scală pentru randamente: $r = m + s\\,T$, cu poziția $m$, scala $s$ și $\\nu$'),
    T('As $\\nu \\to \\infty$, $t(\\nu) \\to N(0,1)$: the Normal distribution is a limiting case', 'Cînd $\\nu \\to \\infty$, $t(\\nu) \\to N(0,1)$: distribuția Normală este un caz limită')))

D.frame(T('Moments and Tails', 'Momente și cozi'), items(
    T('Mean $0$ if $\\nu > 1$; variance $\\nu/(\\nu - 2)$ if $\\nu > 2$; excess kurtosis $6/(\\nu - 4)$ if $\\nu > 4$', 'Media $0$ dacă $\\nu > 1$; varianța $\\nu/(\\nu - 2)$ dacă $\\nu > 2$; excesul de boltire $6/(\\nu - 4)$ dacă $\\nu > 4$'),
    (T('Moments of order $\\nu$ and higher are infinite', 'Momentele de ordin $\\nu$ și mai mare sînt infinite'),
     [T('$\\nu \\le 4$: infinite kurtosis; $\\nu \\le 2$: infinite variance', '$\\nu \\le 4$: boltire infinită; $\\nu \\le 2$: varianță infinită')]),
    (T('\\textbf{Power-law tails}: $P(|T| > x) \\approx c\\,x^{-\\nu}$ for large $x$', '\\textbf{Cozi de tip putere}: $P(|T| > x) \\approx c\\,x^{-\\nu}$ pentru $x$ mare'),
     [T('$c > 0$: a constant; doubling $x$ divides the tail probability by about $2^{\\nu}$', '$c > 0$: o constantă; dublarea lui $x$ împarte probabilitatea din coadă la aproximativ $2^{\\nu}$'),
      T('the Normal tail falls like $e^{-x^2/2}$, much faster than any power', 'coada Normală scade ca $e^{-x^2/2}$, mult mai repede decît orice putere'),
      T('$\\nu$ is also the \\textbf{tail index}, estimated directly in Chapter 5', '$\\nu$ este și \\textbf{tail index}-ul, estimat direct în Capitolul 5')]),
    T('To compare with the Normal distribution at the same variance, use $T\\sqrt{(\\nu - 2)/\\nu}$, which has variance 1', 'Pentru comparația cu distribuția Normală la aceeași varianță, folosiți $T\\sqrt{(\\nu - 2)/\\nu}$, care are varianța 1')))

chart(T('Student-$t$ Densities with Unit Variance', 'Densități Student-$t$ cu varianța 1'), 'sfm_ch2_t_densities', 'SFM_ch2_student_t_fit', [
    T('Smaller $\\nu$: a taller centre and heavier tails; on the log scale the Normal tails drop like a parabola', '$\\nu$ mai mic: centru mai înalt și cozi mai groase; pe scara logaritmică, cozile Normale scad ca o parabolă'),
    T('$P(|X| > 4)$ at unit variance: $t(3)$ @{td.3.p4}\\%, $t(5)$ @{td.5.p4}\\%, $t(10)$ @{td.10.p4}\\%, Normal distribution @{td.n.p4}\\%',
      '$P(|X| > 4)$ la varianța 1: $t(3)$ @{td.3.p4}\\%, $t(5)$ @{td.5.p4}\\%, $t(10)$ @{td.10.p4}\\%, distribuția Normală @{td.n.p4}\\%'),
    T('With $\\nu = 5$, a 4-sigma day is @{td.ratio5} times more likely than under the Normal distribution', 'Cu $\\nu = 5$, o zi de 4 sigma este de @{td.ratio5} de ori mai probabilă decît în cazul distribuției Normale')], h='0.60\\textheight')

D.frame(T('Fitting a Student-$t$ by Maximum Likelihood', 'Estimarea unei distribuții Student-$t$ prin verosimilitate maximă'), items(
    (T('\\textbf{MLE} (maximum likelihood estimation): choose $(\\nu, m, s)$ that maximise the log-likelihood',
       '\\textbf{MLE} (maximum likelihood estimation, estimarea prin verosimilitate maximă): alegem $(\\nu, m, s)$ care maximizează log-verosimilitatea'),
     [T('$\\ell(\\nu, m, s) = \\sum_{t=1}^n \\ln f\\big((r_t - m)/s; \\nu\\big) - n \\ln s$', '$\\ell(\\nu, m, s) = \\sum_{t=1}^n \\ln f\\big((r_t - m)/s; \\nu\\big) - n \\ln s$'),
      T('$f(\\cdot; \\nu)$: the $t(\\nu)$ density; $-n \\ln s$ corrects for rescaling the returns by $s$', '$f(\\cdot; \\nu)$: densitatea $t(\\nu)$; termenul $-n \\ln s$ corectează rescalarea randamentelor cu $s$'),
      T('no closed form: a numerical optimiser (\\texttt{scipy.stats.t.fit} in the notebook)', 'fără formă închisă: un optimizator numeric (\\texttt{scipy.stats.t.fit} în notebook)')]),
    (T('\\textbf{AIC} (Akaike information criterion) \\refAkaike: $\\text{AIC} = 2k - 2\\ell$', '\\textbf{AIC} (Akaike information criterion, criteriul informațional Akaike) \\refAkaike: $\\text{AIC} = 2k - 2\\ell$'),
     [T('$k$ = number of parameters (2 for the Normal distribution, 3 for the $t$); the lower AIC wins', '$k$ = numărul de parametri (2 pentru distribuția Normală, 3 pentru $t$); se preferă modelul cu AIC mai mic'),
      T('$\\Delta\\text{AIC} = \\text{AIC}_{\\text{Normal}} - \\text{AIC}_{t} > 0$ favours the $t$', '$\\Delta\\text{AIC} = \\text{AIC}_{\\text{Normal}} - \\text{AIC}_{t} > 0$ favorizează $t$')]),
    T('Early evidence for stock prices: \\refPraetz, \\refBG', 'Primele rezultate empirice pentru prețurile acțiunilor: \\refPraetz, \\refBG')))

D.frame(T('Fitted Student-$t$: Degrees of Freedom, @{y0}--@{y1}', 'Distribuția Student-$t$ estimată: gradele de libertate, @{y0}--@{y1}'),
        cols(table('lrrr', T('Series', 'Seria') + ' & $\\hat\\nu$ & JB & $\\Delta$AIC', [trow(k) for k in ASSETS], size='footnotesize'), items(
            T('All $\\hat\\nu$ between @{nu.min} (@{nu.min.name}) and @{nu.max} (@{nu.max.name})', 'Toate valorile $\\hat\\nu$ între @{nu.min} (@{nu.min.name}) și @{nu.max} (@{nu.max.name})'),
            T('$\\hat\\nu < 4$ for @{nu.below4} of the @{nassets} series: the fitted model has infinite kurtosis', '$\\hat\\nu < 4$ pentru @{nu.below4} din cele @{nassets} serii: modelul estimat are boltire infinită'),
            T('$\\Delta$AIC is at least @{daic.min}: the $t$ beats the Normal distribution everywhere', '$\\Delta$AIC este cel puțin @{daic.min}: distribuția $t$ este preferată distribuției Normale pentru toate seriile')), wl='0.55', wr='0.41') + ql('SFM_ch2_student_t_fit'),
        'footnotesize')

chart(T('S\\&P 500: Normal vs Student-$t$ Fit', 'S\\&P 500: distribuția Normală și distribuția Student-$t$ estimate'), 'sfm_ch2_t_fit_sp500', 'SFM_ch2_student_t_fit', [
    T('Linear scale: the $t$ ($\\hat\\nu = @{m.sp500.nu}$) follows the tall centre that the Normal distribution misses', 'Scara liniară: distribuția $t$ ($\\hat\\nu = @{m.sp500.nu}$) reproduce centrul înalt, pe care distribuția Normală nu îl reproduce'),
    T('Log scale: the Normal density collapses beyond $\\pm 4\\%$; the $t$ follows the observed tails', 'Scara logaritmică: densitatea Normală scade abrupt dincolo de $\\pm 4\\%$; distribuția $t$ urmează cozile observate')], h='0.62\\textheight')

chart(T('QQ Plots against the Fitted Student-$t$', 'QQ plots față de distribuția Student-$t$ estimată'), 'sfm_ch2_qq_t', 'SFM_ch2_qq_plots', [
    T('The centre and the moderate tails now lie on the line', 'Centrul și cozile moderate se află acum pe dreaptă'),
    T('Extremes: the fitted $t$ expects lower minima than observed: S\\&P 500 $@{qqt.sp.tmin}\\%$ vs $@{qqt.sp.min}\\%$; Bitcoin $@{qqt.btc.tmin}\\%$ vs $@{qqt.btc.min}\\%$',
      'Extremele: distribuția $t$ estimată prevede minime mai mici decît cele observate: S\\&P 500 $@{qqt.sp.tmin}\\%$ față de $@{qqt.sp.min}\\%$; Bitcoin $@{qqt.btc.tmin}\\%$ față de $@{qqt.btc.min}\\%$'),
    T('With $\\hat\\nu$ close to 2 the model overshoots the most extreme quantiles', 'Cu $\\hat\\nu$ apropiat de 2, modelul supraestimează cuantilele cele mai extreme')], h='0.60\\textheight')

D.frame(T('What the Student-$t$ Still Misses', 'Limitele distribuției Student-$t$'), items(
    (T('\\textbf{Symmetry}: the $t$ cannot reproduce the negative skewness', '\\textbf{Simetria}: $t$ nu poate reproduce asimetria negativă'),
     [T('skewed $t$ distributions exist; they are beyond this chapter', 'există distribuții $t$ asimetrice, dar depășesc cadrul acestui capitol')]),
    (T('\\textbf{Independence}: an i.i.d. $t$ model has no volatility clustering', '\\textbf{Independența}: un model $t$ i.i.d. nu are volatility clustering'),
     [T('GARCH models with $t$ errors combine both \\refBollerslev{} (Chapter 9)', 'modelele GARCH cu erori $t$ le combină pe amîndouă \\refBollerslev{} (Capitolul 9)')]),
    (T('\\textbf{Single tail index}: one $\\nu$ for both tails and for the centre', '\\textbf{Un singur tail index}: un singur $\\nu$ pentru ambele cozi și pentru centru'),
     [T('extreme value theory models each tail separately (Chapter 5)', 'teoria valorilor extreme modelează fiecare coadă separat (Capitolul 5)')]),
    T('Still the simplest heavy-tailed model, and a large improvement on the Normal distribution', 'Rămîne cel mai simplu model cu cozi groase și o îmbunătățire mare față de distribuția Normală')))

D.recap(('Student-$t$', 'Student-$t$'), [
    T('One extra parameter $\\nu$ controls the tails; power-law tails, the Normal distribution as $\\nu \\to \\infty$', 'Un parametru în plus, $\\nu$, controlează cozile; cozi de tip putere, distribuția Normală cînd $\\nu \\to \\infty$'),
    T('Fitted $\\hat\\nu$ between @{nu.min0} and @{nu.max0} for daily returns; AIC strongly prefers the $t$', '$\\hat\\nu$ estimat între @{nu.min0} și @{nu.max0} pentru randamentele zilnice; AIC preferă clar $t$'),
    T('It misses skewness and volatility clustering', 'Nu surprinde asimetria și volatility clustering')])

# =============================================================================
# 8. FAPTELE STILIZATE
# =============================================================================
D.section('Stylised Facts of Returns', 'Faptele stilizate ale randamentelor')

D.frame(T('What Is a Stylised Fact?', 'Definiția faptului stilizat'), cols(items(
    T('A \\textbf{stylised fact}: a statistical property shared by many assets, markets and periods \\refCont',
      'Un \\textbf{fapt stilizat}: o proprietate statistică comună multor active, piețe și perioade \\refCont'),
    T('Qualitative, not exact: ``heavy tails\'\', not ``$\\nu = 3.2$\'\'', 'Calitativ, nu exact: „cozi groase”, nu „$\\nu = 3.2$”'),
    T('Rama Cont lists eleven; we check six on our data', 'Rama Cont enumeră unsprezece; noi verificăm șase pe datele noastre'),
    T('A good model of returns must reproduce them; the benchmark i.i.d. Normal model fails most of them',
      'Un model bun al randamentelor trebuie să le reproducă; modelul de referință i.i.d. Normal nu reproduce majoritatea lor'),
    T('Earlier evidence: \\refMandelbrot, \\refFama', 'Primele rezultate empirice: \\refMandelbrot, \\refFama')),
    ph('cont', 'Rama Cont', h='0.42\\textheight'), wl='0.60', wr='0.36'))

D.frame(T('Six Stylised Facts', 'Șase fapte stilizate'), items(
    T('\\textbf{1. Heavy tails}: large moves are far more frequent than under the Normal distribution', '\\textbf{1. Cozi groase}: variațiile mari sînt mult mai frecvente decît prevede distribuția Normală'),
    T('\\textbf{2. Aggregational Gaussianity}: returns over longer horizons are closer to the Normal distribution', '\\textbf{2. Gaussianitate agregată} (aggregational Gaussianity): randamentele pe orizonturi mai lungi sînt mai apropiate de distribuția Normală'),
    T('\\textbf{3. Absence of linear autocorrelation}: past returns barely predict future returns', '\\textbf{3. Absența autocorelației liniare}: randamentele trecute aproape nu au putere de predicție pentru cele viitoare'),
    T('\\textbf{4. Volatility clustering}: large moves follow large moves, of either sign', '\\textbf{4. Volatility clustering}: variațiile mari sînt urmate de variații mari, de orice semn'),
    T('\\textbf{5. Leverage effect}: falls raise future volatility more than rises', '\\textbf{5. Efectul de levier} (leverage effect): scăderile cresc volatilitatea viitoare mai mult decît creșterile'),
    T('\\textbf{6. Gain/loss asymmetry}: large losses are larger and more frequent than large gains', '\\textbf{6. Asimetria cîștig/pierdere}: pierderile mari sînt mai mari și mai frecvente decît cîștigurile mari')))

D.frame(T('Fact 1: Heavy Tails, Counted', 'Faptul 1: cozile groase în cifre'), table(
    'lrrrrrr', T('Series', 'Seria') + ' & $n$ & $>3\\sigma$ & $>4\\sigma$ & $>5\\sigma$ & $>6\\sigma$ & p-value (Poisson, $4\\sigma$)',
    [f'{NAMES[k]} & @{{m.{k}.n}} & @{{sg.{k}.3.obs}} & @{{sg.{k}.4.obs}} & @{{sg.{k}.5.obs}} & @{{sg.{k}.6.obs}} & ${{@{{sg.{k}.4.pp}}}}$' for k in INDICES]
    + [T('Normal distribution, $n = @{m.sp500.n}$', 'Distribuția Normală, $n = @{m.sp500.n}$') + ' & & $@{sg.sp500.3.exp}$ & $@{sg.sp500.4.exp}$ & $@{sg.sp500.5.exp}$ & $@{sg.sp500.6.exp}$ & '],
    size='footnotesize') + items(
    T('$>k\\sigma$: number of days with $|r_t - \\bar r| > k\\,s$; last row: number expected under the Normal distribution', '$>k\\sigma$: numărul zilelor cu $|r_t - \\bar r| > k\\,s$; ultimul rînd: numărul așteptat conform distribuției Normale'),
    T('p-value: probability of at least that many 4-sigma days if the Normal count is Poisson with mean $n \\times @{nt.4.p}$', 'p-value: probabilitatea de a observa cel puțin atîtea zile de 4 sigma, dacă numărul lor urmează distribuția Poisson cu media $n \\times @{nt.4.p}$ (ipoteza de normalitate)'),
    T('S\\&P 500: @{sg.sp.ratio} times more 4-sigma days than the Normal distribution allows', 'S\\&P 500: de @{sg.sp.ratio} de ori mai multe zile de 4 sigma decît permite distribuția Normală')) + ql('SFM_ch2_sigma_days'),
    'footnotesize')

chart(T('Fact 1: 4-Sigma Days in Every Market', 'Faptul 1: zile de 4 sigma pe toate piețele'), 'sfm_ch2_sigma_days', 'SFM_ch2_sigma_days', [
    T('Observed days beyond 4 standard deviations (log scale) against the Normal expectation of $@{ex4.min}$--$@{ex4.max}$ days',
      'Zilele observate dincolo de 4 abateri standard (scară logaritmică), față de numărul așteptat conform distribuției Normale, între $@{ex4.min}$ și $@{ex4.max}$ zile'),
    T('Indices and Bitcoin: one 4-sigma day every @{ev.min}--@{ev.max} days; BVB stocks: @{stk.min}--@{stk.max} days in total',
      'Indicii și Bitcoin: o zi de 4 sigma la fiecare @{ev.min}--@{ev.max} zile; acțiunile BVB: între @{stk.min} și @{stk.max} zile în total'),
    T('Under a fitted Student-$t$, the expected numbers are close to the observed ones: S\\&P 500 @{b4.sp500.expt}, BET @{b4.bet.expt} (Seminar 2, B4)',
      'Cu o distribuție Student-$t$ estimată, numerele așteptate sînt apropiate de cele observate: S\\&P 500 @{b4.sp500.expt}, BET @{b4.bet.expt} (Seminarul 2, B4)')],
      h='0.60\\textheight')

D.frame(T('Fact 2: Aggregational Gaussianity', 'Faptul 2: gaussianitatea agregată'), items(
    (T('$h$-day log return: $r_t(h) = r_t + r_{t-1} + \\dots + r_{t-h+1}$, taken on non-overlapping blocks', 'Randamentul logaritmic pe $h$ zile: $r_t(h) = r_t + r_{t-1} + \\dots + r_{t-h+1}$, calculat pe blocuri disjuncte'),
     [T('if the CLT applies, the distribution of $r_t(h)$ approaches the Normal distribution as $h$ grows', 'dacă se aplică CLT, distribuția lui $r_t(h)$ se apropie de distribuția Normală cînd $h$ crește')]),
    T('The stylised fact: the shape changes with the horizon; it is not the same at all time scales \\refCont', 'Faptul stilizat: forma se schimbă cu orizontul; nu este aceeași la toate scările de timp \\refCont'),
    (T('Price to pay: fewer observations', 'Prețul plătit: mai puține observații'),
     [T('S\\&P 500, @{years.span} years: @{ag.n1} daily, @{ag.n21} monthly and @{ag.n63} quarterly returns', 'S\\&P 500, @{years.span} ani: @{ag.n1} randamente zilnice, @{ag.n21} randamente lunare și @{ag.n63} randamente trimestriale'),
      T('estimates of kurtosis at long horizons are very noisy', 'estimările boltirii pe orizonturi lungi sînt foarte imprecise')])))

chart(T('Fact 2: Kurtosis Falls with the Horizon', 'Faptul 2: boltirea scade cu orizontul'), 'sfm_ch2_aggregation', 'SFM_ch2_cont_stylised_facts', [
    T('Excess kurtosis, daily $\\to$ monthly $\\to$ quarterly: S\\&P 500 @{ag.sp500.1} $\\to$ @{ag.sp500.21} $\\to$ @{ag.sp500.63}; BET @{ag.bet.1} $\\to$ @{ag.bet.21} $\\to$ @{ag.bet.63}; Bitcoin @{ag.btc.1} $\\to$ @{ag.btc.21} $\\to$ @{ag.btc.63}',
      'Excesul de boltire, zilnic $\\to$ lunar $\\to$ trimestrial: S\\&P 500 @{ag.sp500.1} $\\to$ @{ag.sp500.21} $\\to$ @{ag.sp500.63}; BET @{ag.bet.1} $\\to$ @{ag.bet.21} $\\to$ @{ag.bet.63}; Bitcoin @{ag.btc.1} $\\to$ @{ag.btc.21} $\\to$ @{ag.btc.63}'),
    T('The fall is slow and not monotone: one crash month (March 2020) dominates a sample of 200 months', 'Scăderea este lentă și nemonotonă: o singură lună de crah (martie 2020) domină un eșantion de 200 de luni'),
    T('Quarterly returns: JB no longer rejects normality for the BET (p-value $@{ag.bet.63.p}$) and Bitcoin (p-value $@{ag.btc.63.p}$)', 'Randamentele trimestriale: JB nu mai respinge normalitatea pentru BET (p-value $@{ag.bet.63.p}$) și Bitcoin (p-value $@{ag.btc.63.p}$)')],
      h='0.60\\textheight')

chart(T('Fact 2: S\\&P 500, Daily vs Monthly QQ Plots', 'Faptul 2: S\\&P 500, QQ plots pentru randamentele zilnice și lunare'), 'sfm_ch2_qq_horizons', 'SFM_ch2_cont_stylised_facts', [
    T('Monthly returns lie much closer to the line than daily returns', 'Randamentele lunare sînt mult mai aproape de dreaptă decît cele zilnice'),
    T('But monthly returns are not Normal: skewness $@{qh.21.skew}$, excess kurtosis $@{qh.21.k}$, $n = @{qh.21.n}$; the lowest point is March 2020',
      'Dar randamentele lunare nu sînt Normale: asimetria $@{qh.21.skew}$, excesul de boltire $@{qh.21.k}$, $n = @{qh.21.n}$; punctul cel mai de jos este martie 2020'),
    T('Why so slow? Volatility clustering makes daily returns dependent, which slows the CLT', 'Cauza convergenței lente: volatility clustering face randamentele zilnice dependente, ceea ce încetinește convergența din CLT')], h='0.60\\textheight')

D.frame(T('Fact 3: Absence of Linear Autocorrelation', 'Faptul 3: absența autocorelației liniare'), items(
    (T('\\textbf{ACF} (autocorrelation function): $\\rho(h) = \\text{Corr}(r_t, r_{t-h})$, $h = 1, 2, \\dots$', '\\textbf{ACF} (autocorrelation function, funcția de autocorelație): $\\rho(h) = \\text{Corr}(r_t, r_{t-h})$, $h = 1, 2, \\dots$'),
     [T('sample version $\\hat\\rho(h) = \\sum_t (r_t - \\bar r)(r_{t-h} - \\bar r)/\\sum_t (r_t - \\bar r)^2$', 'versiunea de selecție $\\hat\\rho(h) = \\sum_t (r_t - \\bar r)(r_{t-h} - \\bar r)/\\sum_t (r_t - \\bar r)^2$'),
      T('$h$: the lag in days; $\\rho(h) \\in [-1, 1]$; for i.i.d. data, $\\hat\\rho(h) \\approx N(0, 1/n)$: 95\\% band $\\pm 1.96/\\sqrt{n}$', '$h$: lagul, în zile; $\\rho(h) \\in [-1; 1]$; pentru date i.i.d., $\\hat\\rho(h) \\approx N(0, 1/n)$: banda de 95\\% $\\pm 1.96/\\sqrt{n}$')]),
    (T('\\textbf{LB} (Ljung--Box) test \\refLB{} of $H_0$: $\\rho(1) = \\dots = \\rho(m) = 0$', 'Testul \\textbf{LB} (Ljung--Box) \\refLB{} pentru $H_0$: $\\rho(1) = \\dots = \\rho(m) = 0$'),
     [T('$Q(m) = n(n+2)\\sum_{h=1}^m \\hat\\rho(h)^2/(n - h) \\sim \\chi^2(m)$; for $m = 10$, reject if $Q > @{lb.crit}$', '$Q(m) = n(n+2)\\sum_{h=1}^m \\hat\\rho(h)^2/(n - h) \\sim \\chi^2(m)$; pentru $m = 10$, respingem dacă $Q > @{lb.crit}$'),
      T('$m$: the number of lags tested together; $Q$ adds up the squared autocorrelations, so large values mean some $\\rho(h) \\neq 0$', '$m$: numărul de laguri testate împreună; $Q$ adună pătratele autocorelațiilor, deci valorile mari arată că un $\\rho(h) \\neq 0$')]),
    T('Why absent? If returns were predictable, traders would exploit it until the predictability disappears (efficient markets, Chapter 7)',
      'Explicația absenței: dacă randamentele ar fi previzibile, investitorii ar exploata acest lucru pînă cînd previzibilitatea dispare (piețe eficiente, Capitolul 7)')))

chart(T('Facts 3 and 4: ACF of Returns and of Absolute Returns', 'Faptele 3 și 4: ACF a randamentelor și a randamentelor absolute'), 'sfm_ch2_acf', 'SFM_ch2_cont_stylised_facts', [
    T('Left: returns; most autocorrelations are inside or close to the band; S\\&P 500 $\\hat\\rho(1) = @{acf.sp500.r1}$, BET $@{acf.bet.r1}$, Bitcoin $@{acf.btc.r1}$',
      'Stînga: randamentele; majoritatea autocorelațiilor sînt în bandă sau aproape de ea; S\\&P 500 $\\hat\\rho(1) = @{acf.sp500.r1}$, BET $@{acf.bet.r1}$, Bitcoin $@{acf.btc.r1}$'),
    T('Right: absolute returns; all positive, slowly decaying: S\\&P 500 $@{acf.sp500.a1}$ at lag 1, $@{acf.sp500.a10}$ at lag 10, $@{acf.sp500.a50}$ at lag 50',
      'Dreapta: randamentele absolute; toate pozitive, cu scădere lentă: S\\&P 500 $@{acf.sp500.a1}$ la lagul 1, $@{acf.sp500.a10}$ la lagul 10, $@{acf.sp500.a50}$ la lagul 50')],
      h='0.60\\textheight')

D.frame(T('Fact 3: Small but Not Always Zero', 'Faptul 3: autocorelație mică, dar nu întotdeauna nulă'), items(
    (T('Ljung--Box $Q(10)$ for returns: S\\&P 500 @{acf.sp500.lbr}, BET @{acf.bet.lbr}, Bitcoin @{acf.btc.lbr}; critical value @{lb.crit}',
       'Ljung--Box $Q(10)$ pentru randamente: S\\&P 500 @{acf.sp500.lbr}, BET @{acf.bet.lbr}, Bitcoin @{acf.btc.lbr}; valoarea critică @{lb.crit}'),
     [T('formally, some autocorrelation is significant', 'formal, unele autocorelații sînt semnificative')]),
    (T('Two reasons not to over-read this', 'Două motive pentru a nu exagera importanța rezultatului'),
     [T('the values are economically small: $|\\hat\\rho(h)| \\le @{acf.maxabs}$, so at most @{acf.r2}\\% of the variance is explained', 'valorile sînt mici din punct de vedere economic: $|\\hat\\rho(h)| \\le @{acf.maxabs}$, deci se explică cel mult @{acf.r2}\\% din varianță'),
      T('the band $\\pm 1.96/\\sqrt{n}$ assumes i.i.d. returns; with volatility clustering it is too narrow, so ``significant\'\' is found too often',
        'banda $\\pm 1.96/\\sqrt{n}$ presupune randamente i.i.d.; cu volatility clustering este prea îngustă, deci „semnificativ” apare prea des')]),
    T('The S\\&P 500 value at lag 1 comes mostly from March 2020, when large falls and rebounds alternated: without 20 February--30 April 2020, $\\hat\\rho(1) = @{acf.sp500.ex}$', 'Valoarea S\\&P 500 la lagul 1 provine mai ales din martie 2020, cînd scăderile mari și revenirile au alternat: fără 20 februarie--30 aprilie 2020, $\\hat\\rho(1) = @{acf.sp500.ex}$'),
    T('Robust tests of predictability (variance ratios): Chapter 7', 'Teste robuste ale previzibilității (testele raportului varianțelor): Capitolul 7')))

D.frame(T('Fact 4: Volatility Clustering', 'Faptul 4: volatility clustering'), cols(items(
    T('``Large changes tend to be followed by large changes, of either sign, and small changes tend to be followed by small changes\'\' \\refMandelbrot',
      '„Variațiile mari tind să fie urmate de variații mari, de orice semn, iar variațiile mici de variații mici” \\refMandelbrot'),
    T('The sign of tomorrow\'s return is unpredictable; its size is not', 'Semnul randamentului de mîine este imprevizibil; mărimea lui nu este'),
    (T('Evidence: ACF of $|r_t|$ positive for 50 lags', 'Dovezi: ACF a lui $|r_t|$ pozitivă pe 50 de laguri'),
     [T('Ljung--Box $Q(10)$ of $|r_t|$: S\\&P 500 @{acf.sp500.lba}, BET @{acf.bet.lba}, Bitcoin @{acf.btc.lba}', 'Ljung--Box $Q(10)$ pentru $|r_t|$: S\\&P 500 @{acf.sp500.lba}, BET @{acf.bet.lba}, Bitcoin @{acf.btc.lba}')]),
    T('So returns are uncorrelated but \\textbf{not independent}', 'Deci randamentele sînt necorelate, dar \\textbf{nu independente}'),
    T('Models: ARCH and GARCH (Chapter 9)', 'Modele: ARCH și GARCH (Capitolul 9)')),
    ph('mandelbrot', 'Benoit Mandelbrot (1924--2010)', h='0.42\\textheight'), wl='0.62', wr='0.34'), 'footnotesize')

chart(T('Fact 4: Calm and Turbulent Periods', 'Faptul 4: perioade calme și perioade agitate'), 'sfm_ch2_clustering', 'SFM_ch2_cont_stylised_facts', [
    T('Turbulent periods: 2010--2011 (euro area debt crisis), December 2018 for the BET, March 2020 everywhere, April 2025', 'Perioade agitate: 2010--2011 (criza datoriilor din zona euro), decembrie 2018 pentru BET, martie 2020 peste tot, aprilie 2025'),
    T('Calm years in between: the volatility of a single day depends on the period it falls in', 'Între ele, ani calmi: volatilitatea unei zile depinde de perioada în care se află')], h='0.62\\textheight')

D.frame(T('Fact 5: The Leverage Effect', 'Faptul 5: efectul de levier'), items(
    (T('Measure: $L(k) = \\text{Corr}(r_t, |r_{t+k}|)$, $k = 1, 2, \\dots$', 'Măsura: $L(k) = \\text{Corr}(r_t, |r_{t+k}|)$, $k = 1, 2, \\dots$'),
     [T('$L(k) < 0$: a fall today is followed by larger moves than a rise of the same size', '$L(k) < 0$: o scădere azi este urmată de variații mai mari decît o creștere de aceeași mărime')]),
    (T('Explanation by financial leverage \\refChristie', 'Explicația prin efectul de levier financiar \\refChristie'),
     [T('when the share price falls, the debt-to-equity ratio rises, so equity becomes riskier', 'cînd prețul acțiunii scade, raportul datorii/capital propriu crește, deci capitalul propriu devine mai riscant')]),
    T('A second explanation, volatility feedback: higher expected volatility raises required returns and lowers prices', 'O a doua explicație, volatility feedback: o volatilitate așteptată mai mare mărește randamentul cerut și reduce prețurile'),
    T('Models with an asymmetric response (EGARCH, GJR-GARCH): Chapter 9', 'Modele cu răspuns asimetric (EGARCH, GJR-GARCH): Capitolul 9')))

chart(T('Fact 5: Leverage Correlations', 'Faptul 5: corelațiile de levier'), 'sfm_ch2_leverage', 'SFM_ch2_cont_stylised_facts', [
    T('Stock indices: $L(k) < 0$ for most of the first 20 lags; mean of $L(1..5)$: S\\&P 500 $@{lev.sp500.m5}$, DAX $@{lev.dax.m5}$, BET $@{lev.bet.m5}$',
      'Indicii bursieri: $L(k) < 0$ pentru majoritatea primelor 20 de laguri; media $L(1..5)$: S\\&P 500 $@{lev.sp500.m5}$, DAX $@{lev.dax.m5}$, BET $@{lev.bet.m5}$'),
    T('Bitcoin: $L(1) = @{lev.btc.l1}$, then inside the band: no firm has debt behind Bitcoin, so the leverage mechanism is absent',
      'Bitcoin: $L(1) = @{lev.btc.l1}$, apoi în interiorul benzii: în spatele Bitcoin nu există o firmă cu datorii, deci mecanismul de levier lipsește')], h='0.60\\textheight')

chart(T('Fact 6: Gain/Loss Asymmetry', 'Faptul 6: asimetria cîștig/pierdere'), 'sfm_ch2_gain_loss', 'SFM_ch2_sigma_days', [
    T('Left: a 1-in-100 loss is larger than a 1-in-100 gain for the indices: BET @{gl.bet.q01}\\% vs @{gl.bet.q99}\\%, S\\&P 500 @{gl.sp500.q01}\\% vs @{gl.sp500.q99}\\%',
      'Stînga: la indici, pierderea depășită într-o zi din 100 este mai mare decît cîștigul depășit într-o zi din 100: BET @{gl.bet.q01}\\% față de @{gl.bet.q99}\\%, S\\&P 500 @{gl.sp500.q01}\\% față de @{gl.sp500.q99}\\%'),
    T('Right: days below $-4\\sigma$ vs above $+4\\sigma$: BET @{gl.bet.d} vs @{gl.bet.u}, S\\&P 500 @{gl.sp500.d} vs @{gl.sp500.u}, Bitcoin @{gl.btc.d} vs @{gl.btc.u}',
      'Dreapta: zile sub $-4\\sigma$ față de peste $+4\\sigma$: BET @{gl.bet.d} față de @{gl.bet.u}, S\\&P 500 @{gl.sp500.d} față de @{gl.sp500.u}, Bitcoin @{gl.btc.d} față de @{gl.btc.u}'),
    T('Exception: SNN, with positive skewness and a larger right quantile', 'Excepție: SNN, cu asimetrie pozitivă și cuantila din dreapta mai mare')], h='0.60\\textheight')

FT = N['facts']
for k in ASSETS:
    f = FT[k]
    V.put(f'ft.{k}.r1', f['rho1'], 2)
    V.put(f'ft.{k}.a10', f['rho10_abs'], 2)
    V.put(f'ft.{k}.lev', f['lev5'], 2)
    V.put(f'ft.{k}.km', f['exkurt_m'], 1)
    V.put(f'ft.{k}.gl', f['gl_ratio'], 2)
D.frame(T('Six Facts, Eight Series', 'Șase fapte, opt serii'), table(
    'lrrrrrr', T('Series', 'Seria') + ' & ' + T('Exc. kurt.', 'Exces bolt.') + ' & ' + T('Exc. kurt., 21 days', 'Exces bolt., 21 zile') + ' & $\\hat\\rho_r(1)$ & $\\hat\\rho_{|r|}(10)$ & $\\bar L(1..5)$ & $|q_{0.01}|/q_{0.99}$',
    [f'{NAMES[k]} & $@{{m.{k}.exkurt}}$ & $@{{ft.{k}.km}}$ & $@{{ft.{k}.r1}}$ & $@{{ft.{k}.a10}}$ & $@{{ft.{k}.lev}}$ & $@{{ft.{k}.gl}}$' for k in ASSETS],
    size='scriptsize') + items(
    T('Columns: excess kurtosis of daily and 21-day returns; ACF of returns at lag 1 and of $|r_t|$ at lag 10; mean $L(k)$ over lags 1--5; 1\\% over 99\\% quantile',
      'Coloanele: excesul de boltire al randamentelor zilnice și pe 21 de zile; ACF a randamentelor la lagul 1 și a lui $|r_t|$ la lagul 10; media $L(k)$ pe lagurile 1--5; cuantila de 1\\% împărțită la cea de 99\\%'),
    T('Facts 1, 3 and 4 hold for every series; fact 2 holds with noise; facts 5 and 6 hold for the indices and most BVB stocks, weakly or not for Bitcoin and SNN',
      'Faptele 1, 3 și 4 sînt valabile pentru toate seriile; faptul 2, cu estimări imprecise; faptele 5 și 6, pentru indici și majoritatea acțiunilor BVB, slab sau deloc pentru Bitcoin și SNN'),
    T('A comparison of the return distributions of cryptocurrencies and traditional assets: \\refPeleCrypto', 'O comparație între distribuțiile randamentelor criptomonedelor și cele ale activelor tradiționale: \\refPeleCrypto')) + ql('SFM_ch2_cont_stylised_facts'),
    'footnotesize')

D.frame(T('What the Facts Mean for Models', 'Implicațiile faptelor stilizate pentru modele'), table(
    'lcccccc', T('Model', 'Model') + ' & 1 & 2 & 3 & 4 & 5 & 6',
    [T('i.i.d. Normal', 'i.i.d. Normal') + ' & -- & ' + T('trivially', 'trivial') + ' & \\checkmark & -- & -- & --',
     T('i.i.d. Student-$t$', 'i.i.d. Student-$t$') + ' & \\checkmark & \\checkmark & \\checkmark & -- & -- & --',
     T('GARCH with $t$ errors (Ch.~9)', 'GARCH cu erori $t$ (cap.~9)') + ' & \\checkmark & \\checkmark & \\checkmark & \\checkmark & -- & --',
     T('EGARCH, GJR-GARCH (Ch.~9)', 'EGARCH, GJR-GARCH (cap.~9)') + ' & \\checkmark & \\checkmark & \\checkmark & \\checkmark & \\checkmark & (\\checkmark)'],
    size='footnotesize') + items(
    T('Columns: the six facts in the order of this section; (\\checkmark): only for returns over several days, unless the errors are skewed', 'Coloanele: cele șase fapte în ordinea din această secțiune; (\\checkmark): doar pentru randamente pe mai multe zile, dacă erorile nu sînt asimetrice'),
    T('Risk measures (VaR 1\\%, ES 2.5\\%, Chapter 10) inherit the quality of the model: a Normal VaR is too low in crises',
      'Măsurile de risc (VaR 1\\%, ES 2,5\\%, Capitolul 10) moștenesc calitatea modelului: un VaR Normal este prea mic în crize'),
    T('A related summary of tail risk, information entropy: \\refPeleEntropy', 'O măsură înrudită a riscului de coadă, entropia informațională: \\refPeleEntropy')))

D.recap(('Stylised Facts', 'Fapte stilizate'), [
    T('Heavy tails: 4-sigma days every few months instead of once in decades', 'Cozi groase: o zi de 4 sigma la fiecare cîteva luni, în loc de una la cîteva decenii'),
    T('Returns are nearly uncorrelated, but their size is strongly autocorrelated', 'Randamentele sînt aproape necorelate, dar mărimea lor este puternic autocorelată'),
    T('Longer horizons are closer to the Normal distribution, slowly', 'Pe orizonturi mai lungi, randamentele se apropie lent de distribuția Normală'),
    T('Stock indices: falls raise volatility, and losses are larger than gains', 'Indicii bursieri: scăderile cresc volatilitatea, iar pierderile sînt mai mari decît cîștigurile')])

# =============================================================================
# 9. AI PENTRU DESCOPERIRE ȘTIINȚIFICĂ
# =============================================================================
D.section('AI for Scientific Discovery', 'AI pentru descoperire științifică')

side(T('An Open Question: Are Bitcoin\'s Tails Getting Thinner?', 'O întrebare deschisă: devin mai subțiri cozile Bitcoin?'), 'sfm_ch2_tails_by_year', 'SFM_ch2_tails_by_year', [
    T('As a market matures (more traders, futures, ETFs), its tails might thin out', 'Pe măsură ce o piață se maturizează (mai mulți participanți, futures, ETF-uri), cozile ei s-ar putea subția'),
    T('Bitcoin, @{yr.first}--@{yr.last}: yearly $\\hat\\nu$ between @{yr.numin} and @{yr.numax}; rank correlation with the year $@{yr.rho}$ (p-value $@{yr.p}$)',
      'Bitcoin, @{yr.first}--@{yr.last}: $\\hat\\nu$ anual între @{yr.numin} și @{yr.numax}; corelația rangurilor cu anul $@{yr.rho}$ (p-value $@{yr.p}$)'),
    T('Why it is open: @{yr.n} noisy yearly estimates, one year (2020, excess kurtosis @{yr.k2020}) dominated by one day', 'De ce rămîne deschisă: @{yr.n} estimări anuale imprecise, un an (2020, exces de boltire @{yr.k2020}) dominat de o singură zi'),
    T('AI tools can speed up such a study; they do not replace checking it \\refWang', 'Instrumentele AI pot accelera un astfel de studiu; nu înlocuiesc verificarea lui \\refWang')], w=0.50, h='0.72\\textheight')

D.frame(T('How AI Could Help', 'Contribuția posibilă a AI'), items(
    T('\\textbf{Literature}: find studies of tail indices of crypto assets and summarise their methods', '\\textbf{Literatura}: găsirea studiilor despre tail index-urile activelor cripto și rezumarea metodelor lor'),
    T('\\textbf{Code}: draft rolling-window estimates of $\\nu$ and of the excess kurtosis', '\\textbf{Cod}: o primă versiune a estimărilor pe ferestre mobile pentru $\\nu$ și excesul de boltire'),
    T('\\textbf{Robustness}: propose other windows, other tail measures (Chapter 5), other crypto assets', '\\textbf{Robustețe}: propunerea altor ferestre, altor măsuri ale cozilor (Capitolul 5), altor active cripto'),
    (T('Example prompt', 'Exemplu de prompt'),
     [T('\\aiprompt{Write a Python function that fits a Student-t distribution to daily Bitcoin log returns on rolling 365-day windows and returns the degrees of freedom with a bootstrap 95\\% interval.}',
        '\\aiprompt{Write a Python function that fits a Student-t distribution to daily Bitcoin log returns on rolling 365-day windows and returns the degrees of freedom with a bootstrap 95\\% interval.}')])))

D.frame(T('What to Check', 'Verificări necesare'), items(
    T('Data: the same price column and calendar for every year; errors among the largest moves', 'Datele: aceeași coloană de preț și același calendar pentru fiecare an; erori printre cele mai mari variații'),
    T('Estimation: did the optimiser converge? $\\hat\\nu$ near 2 or above 30 needs a second look', 'Estimarea: a convers optimizatorul? $\\hat\\nu$ aproape de 2 sau peste 30 trebuie reverificat'),
    T('Inference: one $\\hat\\nu$ per year is noisy; report intervals, not only point estimates', 'Inferența: o singură estimare $\\hat\\nu$ pe an este imprecisă; raportați intervale, nu doar estimări punctuale'),
    T('Volatility clustering: thinner yearly tails may only mean calmer years (Chapter 9)', 'Volatility clustering: cozi anuale mai subțiri pot însemna doar ani mai calmi (Capitolul 9)'),
    T('References: every cited paper must exist; check the DOI', 'Referințele: fiecare lucrare citată trebuie să existe; verificați DOI-ul')))

D.frame(T('Project Seed', 'Idee de proiect'), items(
    (T('\\textbf{Question}: do the tails of crypto assets thin out as their markets mature?', '\\textbf{Întrebarea}: se subțiază cozile activelor cripto pe măsură ce piețele lor se maturizează?'),
     [T('data: Bitcoin, Ethereum and Solana from the course data; S\\&P 500 as a benchmark', 'date: Bitcoin, Ethereum și Solana din datele cursului; S\\&P 500 ca reper')]),
    (T('Steps', 'Pași'),
     [T('estimate $\\hat\\nu$ and the excess kurtosis on rolling one-year windows, with bootstrap intervals', 'estimați $\\hat\\nu$ și excesul de boltire pe ferestre mobile de un an, cu intervale bootstrap'),
      T('test for a trend; repeat on returns divided by a rolling volatility', 'testați existența unui trend; repetați pe randamente împărțite la o volatilitate mobilă'),
      T('compare the count of 4-sigma days per year with the S\\&P 500', 'comparați numărul zilelor de 4 sigma pe an cu S\\&P 500')]),
    T('Deliverable: one table, one chart, and a paragraph on what the data can and cannot show', 'Livrabile: un tabel, un grafic și un paragraf despre ce pot și ce nu pot arăta datele'),
    T('Declare any AI use, and list the errors of the AI that you corrected', 'Declarați orice folosire a AI și listați erorile AI pe care le-ați corectat')))

# =============================================================================
# REZUMAT
# =============================================================================
D.section('Summary', 'Rezumat')

D.frame(T('Key Takeaways', 'Idei principale'), items(
    T('The Normal distribution: two parameters, thin tails, closed under addition', 'Distribuția Normală: doi parametri, cozi subțiri, închisă la adunare'),
    T('Normal log returns give lognormal prices; mean above median by the volatility drag', 'Randamentele logaritmice Normale conduc la prețuri lognormale; media depășește mediana prin volatility drag'),
    T('The CLT needs independence, identical distribution and finite variance', 'CLT cere independență, distribuție identică și varianță finită'),
    T('Daily returns: mostly negative skewness, excess kurtosis @{ku.min0}--@{ku.max0}, JB rejects, QQ plots bend at both ends', 'Randamentele zilnice: în general asimetrie negativă, exces de boltire @{ku.min0}--@{ku.max0}, JB respinge, QQ plots se curbează la ambele capete'),
    T('The Student-$t$ with $\\hat\\nu$ between @{nu.min0} and @{nu.max0} fits the tails much better, but is symmetric and i.i.d.', 'Student-$t$ cu $\\hat\\nu$ între @{nu.min0} și @{nu.max0} descrie mult mai bine cozile, dar este simetrică și i.i.d.'),
    T('Six stylised facts: heavy tails, aggregational Gaussianity, no linear autocorrelation, volatility clustering, leverage, gain/loss asymmetry',
      'Șase fapte stilizate: cozi groase, gaussianitate agregată, fără autocorelație liniară, volatility clustering, efect de levier, asimetrie cîștig/pierdere')))

D.frame(T('Key Formulas', 'Formule de reținut'), table(
    'll', T('\\textbf{Quantity}', '\\textbf{Mărimea}') + ' & ' + T('\\textbf{Formula}', '\\textbf{Formula}'),
    [T('Normal PDF', 'Densitatea Normală') + ' & $f(x) = (\\sigma\\sqrt{2\\pi})^{-1}\\exp\\big(-(x - \\mu)^2/(2\\sigma^2)\\big)$',
     T('Lognormal moments', 'Momentele lognormale') + ' & ' + T('median $e^m$, mean $e^{m + s^2/2}$', 'mediana $e^m$, media $e^{m + s^2/2}$'),
     'CLT & $\\sqrt{n}(\\bar X_n - \\mu)/\\sigma \\xrightarrow{d} N(0,1)$',
     T('Skewness, excess kurtosis', 'Asimetria, excesul de boltire') + ' & $S = m_3/m_2^{3/2}$, \\quad $K = m_4/m_2^2 - 3$',
     'Jarque--Bera & $\\text{JB} = \\frac{n}{6}(S^2 + K^2/4) \\sim \\chi^2(2)$',
     'Student-$t(\\nu)$ & ' + T('variance $\\nu/(\\nu - 2)$, excess kurtosis $6/(\\nu - 4)$', 'varianța $\\nu/(\\nu - 2)$, excesul de boltire $6/(\\nu - 4)$'),
     'AIC & $2k - 2\\ell$',
     'ACF, Ljung--Box & $\\hat\\rho(h)$, \\quad $Q(m) = n(n+2)\\sum_{h=1}^m \\hat\\rho(h)^2/(n - h)$',
     T('Leverage', 'Levier') + ' & $L(k) = \\text{Corr}(r_t, |r_{t+k}|)$'],
    size='footnotesize'))

D.frame(T('Check Yourself', 'Autoevaluare'), items(
    (T('\\textbf{Question}: daily returns are $N(0.05\\%, 1\\%^2)$; how many standard deviations is a $-3\\%$ day?',
       '\\textbf{Întrebare}: randamentele zilnice sînt $N(0.05\\%, 1\\%^2)$; la cîte abateri standard se află o zi de $-3\\%$?'),
     [T('\\textbf{Answer}: $z = (-3 - 0.05)/1 = @{cy.z}$', '\\textbf{Răspuns}: $z = (-3 - 0.05)/1 = @{cy.z}$')]),
    (T('\\textbf{Question}: $n = 2500$, $S = -0.5$, $K = 6$; what is JB?', '\\textbf{Întrebare}: $n = 2500$, $S = -0.5$, $K = 6$; cît este JB?'),
     [T('\\textbf{Answer}: $\\frac{2500}{6}(0.25 + 9) = @{cy.jb}$; normality is rejected', '\\textbf{Răspuns}: $\\frac{2500}{6}(0.25 + 9) = @{cy.jb}$; normalitatea este respinsă')]),
    (T('\\textbf{Question}: an annual log return is $N(5\\%, 20\\%^2)$; what are the median and the mean of 100 invested after one year?',
       '\\textbf{Întrebare}: un randament logaritmic anual este $N(5\\%, 20\\%^2)$; care sînt mediana și media valorii unei investiții de 100 după un an?'),
     [T('\\textbf{Answer}: median $100e^{0.05} = @{cy.med}$, mean $100e^{0.05 + 0.02} = @{cy.mean}$', '\\textbf{Răspuns}: mediana $100e^{0.05} = @{cy.med}$, media $100e^{0.05 + 0.02} = @{cy.mean}$')]),
    T('Next: Chapter 3, $\\alpha$-stable distributions', 'Urmează: Capitolul 3, distribuții $\\alpha$-stabile')))

D.references(BIB, per=11)

if __name__ == '__main__':
    D.write(V)
