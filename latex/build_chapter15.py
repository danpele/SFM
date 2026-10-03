r"""
build_chapter15.py -- Capitolul 15 (Risc sistemic), EN + RO dintr-o singură sursă
================================================================================
Text ⟦english||română⟧; cifrele @{cheie} vin din Quantlets/Ch_15/ch15_numbers.json (generate_all_charts.py) sau
sînt calculate aici, în Python, pentru exemplele lucrate. Nicio cifră nu este scrisă de mînă.
Convenția nivelului: VaR 1%, CoVaR 1%, MES 5% (nivelul = probabilitatea cozii); pierderile sînt pozitive.
Ieșire:
  EN/Courses/chapter15_systemic_risk.tex
  RO/Cursuri/capitol15_risc_sistemic.tex
Rulare:
  python3 Quantlets/Ch_15/generate_all_charts.py
  python3 latex/build_chapter15.py && python3 latex/sfm_build.py compile 15
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_build import Deck, cols, items, table, photo   # noqa: E402
from ch15_common import ALL, BIB, EU, NAMES, QLURL, REFS, RO, US, T, load, values   # noqa: E402

N = load()
V = values(N)
D = Deck(15, 'lecture', refs=REFS)
C = 'https://commons.wikimedia.org/wiki/File:'


def ql(folder):
    return f'\\sfmquantlet{{Ch_15}}{{{folder}}}'


def chart(title, fig, folder, bullets, h='0.64\\textheight', size='footnotesize'):
    body = (f'\\begin{{center}}\n\\includegraphics[width=0.97\\textwidth,height={h},keepaspectratio]{{{fig}.pdf}}\n'
            f'\\end{{center}}\n\\vspace{{-0.25cm}}\n' + items(*bullets) + '\n' + ql(folder))
    D.frame(title, body, size)


PH = {
    'lehman': ('ch0_lehman_2008.jpg', C + 'Lehman_Brothers-NYC-20080915.jpg',
               '⟦Photo||Foto⟧: Robert Scoble (2008); CC BY 2.0; Wikimedia Commons'),
    'nrock': ('ch15_northern_rock_2007.jpg', C + 'Northern_Rock_Queue.jpg',
              '⟦Photo||Foto⟧: Dominic Alves (2007); CC BY 2.0; Wikimedia Commons'),
    'ecb': ('ch15_ecb_frankfurt_2015.jpg', C + 'Seat_of_the_European_Central_Bank_and_Frankfurt_Skyline_at_dawn_20150422_1.jpg',
            '⟦Photo||Foto⟧: D\\-XR (2015); CC BY-SA 4.0; Wikimedia Commons'),
    'fidi': ('ch0_fidi_march_2020.jpg', C + 'Subdued_FiDi_(50063555551).jpg',
             '⟦Photo||Foto⟧: Billie Grace Ward (2020); CC BY 2.0; Wikimedia Commons'),
    'svb': ('ch15_svb_hq_2023.jpg', C + '3003_West_Tasman_Drive_entrance_2,_Santa_Clara,_California.jpg',
            '⟦Photo||Foto⟧: Minh Nguyen (2023); CC BY-SA 4.0; Wikimedia Commons'),
    'cs': ('ch15_credit_suisse_zurich_2013.jpg', C + 'Credit_Suisse_Z\\%C3\\%BCrich.jpg',
           '⟦Photo||Foto⟧: Thomas Wolf, www.foto-tw.de (2013); CC BY-SA 3.0 DE; Wikimedia Commons'),
    'adrian': ('ch15_adrian_2025.jpg', C + 'Tobias_Adrian_2025.jpg',
               '⟦Photo||Foto⟧: Chianoo Adrian (2025); CC BY 4.0; Wikimedia Commons'),
    'brunn': ('ch15_brunnermeier_2015.jpg', C + 'Markus_Konrad_Brunnermeier.jpg',
              '⟦Photo||Foto⟧: Iitkgpswapnil (2015); CC BY-SA 4.0; Wikimedia Commons'),
    'acharya': ('ch15_acharya_2014.jpg', C + 'Viral_Acharya_(2014).jpg',
                '⟦Photo||Foto⟧: MeJudice (2014); CC BY 3.0; Wikimedia Commons'),
    'engle': ('ch9_engle_2022.jpg', C + '0603-Kraneshares_KRBN-RobertEngle-JonDemske-16_(cropped).jpg',
              '⟦Photo||Foto⟧: Jon Demske (2022); CC BY-SA 4.0; Wikimedia Commons'),
    'bis': ('ch10_bis_tower_2014.jpg', C + 'Basel_Committee_on_Banking_Supervision_-_BCBS.jpg',
            '⟦Photo||Foto⟧: Taxiarchos228 (2014); CC BY-SA 4.0; Wikimedia Commons'),
    'bnr': ('ch15_bnr_palace_2018.jpg', C + 'Bucuresti,_Romania._BANCA_NATIONALA_A_ROMANIEI_(2)_(B-II-m-A-19023).jpg',
            '⟦Photo||Foto⟧: Britchi Mirela (2018); CC BY-SA 4.0; Wikimedia Commons'),
}


def ph(key, cap, h='0.50\\textheight'):
    f, url, cred = PH[key]
    return photo(f, cap, url, cred, h=h)


# =============================================================================
# EXEMPLE LUCRATE (calculate aici)
# =============================================================================
Q = N['qr']
# MES on a toy table of 10 days: market and bank returns (%), level 20% -> the two worst market days
TOYM = [-4.0, 1.2, -0.5, 2.1, -6.0, 0.3, -1.1, 0.8, -2.2, 1.5]
TOYB = [-6.5, 1.8, -0.2, 2.9, -9.0, 0.1, -2.0, 1.1, -3.1, 2.4]
V.put('toy.mes', -(TOYB[4] + TOYB[0]) / 2, 2)
V.put('toy.es', -(TOYM[4] + TOYM[0]) / 2, 2)
# LRMES and SRISK, worked example: MES at -2% = 3%, D = 900, W = 100, k = 8%
lr_ex = 1 - math.exp(-18 * 0.03)
V.put('ex.lr', 100 * lr_ex, 1)
V.put('ex.exp', math.exp(-0.54), 3)
V.put('ex.kd', 0.08 * 900, 0)
V.put('ex.cap', 0.92 * (1 - lr_ex) * 100, 1)
V.put('ex.srisk', 0.08 * 900 - 0.92 * (1 - lr_ex) * 100, 1)
V.put('ex.Lstar', 1 + 0.92 * (1 - lr_ex) / 0.08, 1)
# Forbes-Rigobon: the correction step by step
F = N['fr']
V.put('fr.inside', 1 + F['delta'] * (1 - F['rho_crisis'] ** 2), 1)
V.put('fr.r2', 1 - F['rho_crisis'] ** 2, 2)
# ranges and extremes quoted in the text (computed, never typed)
CT, MT = N['covar'], N['mes']
dc_us = [CT[k]['dcovar'] for k in US]
dc_eu = [CT[k]['dcovar'] for k in EU]
V.put('rg.dus.lo', min(dc_us), 1)
V.put('rg.dus.hi', max(dc_us), 1)
V.put('rg.deu.lo', min(dc_eu), 1)
V.put('rg.deu.hi', max(dc_eu), 1)
V.raw('top.var', NAMES[max(ALL, key=lambda k: CT[k]['var_i'])])
top3v = sorted(ALL, key=lambda k: -CT[k]['var_i'])[:3]
low3d = sorted(ALL, key=lambda k: CT[k]['dcovar'])[:3]
both = [k for k in top3v if k in low3d]
assert len(both) == 2, both
V.raw('vd.b1', NAMES[both[0]])
V.raw('vd.b2', NAMES[both[1]])
V.raw('top.dcov', NAMES[max(ALL, key=lambda k: CT[k]['dcovar'])])
assert sorted(ALL, key=lambda k: -MT[k]['mes5'])[:2] == ['C', 'INGA'] and min(ALL, key=lambda k: MT[k]['mes5']) == 'HSBA'
V.put('rg.lr.lo', 100 * min(MT[k]['lrmes'] for k in ALL), 0)
V.put('rg.lr.hi', 100 * max(MT[k]['lrmes'] for k in ALL), 0)
V.put('rg.L.lo', min(MT[k]['Lstar'] for k in ALL), 1)
V.put('rg.L.hi', max(MT[k]['Lstar'] for k in ALL), 1)
pick = N['srisk_pick']
V.raw('sr.hi', NAMES[pick[0]])
V.raw('sr.lo', NAMES[pick[2]])
V.put('sr.hiL', MT[pick[0]]['Lstar'], 1)
V.put('sr.loL', MT[pick[2]]['Lstar'], 1)
V.put('sr.hi15', 100 * (0.08 * 14 - 0.92 * (1 - MT[pick[0]]['lrmes'])), 0)
V.put('sr.lo15', 100 * (0.08 * 14 - 0.92 * (1 - MT[pick[2]]['lrmes'])), 0)
TO = N['dy']['to']
V.raw('dy.topto', NAMES[max(TO, key=TO.get)])
V.raw('dy.lowto', NAMES[min(TO, key=TO.get)])
mc = N['mes_crisis']['banks']
V.raw('mc.worst', NAMES[min(mc, key=lambda k: mc[k]['ret'])])
V.raw('mc.best', NAMES[max(mc, key=lambda k: mc[k]['ret'])])

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's Question and Route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: when does the loss of one bank become a loss of the whole financial system, and how do we measure it from market data?',
       '\\textbf{Întrebarea}: cînd devine pierderea unei bănci o pierdere a întregului sistem financiar și cum o măsurăm din datele de piață?'),
     [T('Chapter 10 measured the risk of one position (VaR 1\\%, ES 2.5\\%); this chapter measures the risk that one institution adds to the system',
        'Capitolul 10 a măsurat riscul unei poziții (VaR 1\\%, ES 2,5\\%); acest capitol măsoară riscul pe care o instituție îl adaugă sistemului')]),
    (T('\\textbf{Route} of the chapter', '\\textbf{Traseul} capitolului'),
     [T('what systemic risk is: contagion, common exposures, fire sales, too big and too interconnected to fail', 'ce este riscul sistemic: contagiune, expuneri comune, vînzări forțate, instituții prea mari sau prea interconectate pentru a fi lăsate să falimenteze'),
      T('four crises in the data: 2008, the euro area in 2011--2012, March 2020, March 2023', 'patru crize în date: 2008, zona euro în 2011--2012, martie 2020, martie 2023'),
      T('measures: correlation and tail dependence, CoVaR and $\\Delta$CoVaR, MES and SRISK, networks', 'măsuri: corelația și dependența în cozi, CoVaR și $\\Delta$CoVaR, MES și SRISK, rețele'),
      T('macroprudential policy: Basel III buffers, the ESRB, the BNR', 'politica macroprudențială: amortizoarele Basel III, ESRB, BNR')])))

D.frame(T('Learning Outcomes', 'Rezultatele învățării'), items(
    T('Explain the channels of systemic risk and recognise them in the crises of 2008, 2011--2012, 2020 and 2023',
      'Explicați canalele riscului sistemic și recunoașteți-le în crizele din 2008, 2011--2012, 2020 și 2023'),
    T('Estimate CoVaR 1\\% and $\\Delta$CoVaR by linear quantile regression and interpret them next to VaR 1\\%',
      'Estimați CoVaR 1\\% și $\\Delta$CoVaR prin regresie cuantilică liniară și interpretați-le alături de VaR 1\\%'),
    T('Compute MES 5\\%, LRMES and SRISK, and explain the role of leverage', 'Calculați MES 5\\%, LRMES și SRISK și explicați rolul efectului de levier'),
    T('Distinguish contagion from interdependence (the Forbes--Rigobon correction) and read a tail-dependence estimate', 'Distingeți contagiunea de interdependență (corecția Forbes--Rigobon) și interpretați o estimare a dependenței în cozi'),
    T('Read a Granger-causality network and a Diebold--Yilmaz connectedness table', 'Interpretați o rețea de cauzalitate Granger și un tabel de conectivitate Diebold--Yilmaz'),
    T('Name the macroprudential tools of Basel III and the institutions that apply them in the EU and in Romania', 'Numiți instrumentele macroprudențiale din Basel III și instituțiile care le aplică în UE și în România')))

D.frame(T('Reading and Tools', 'Bibliografie și instrumente'), items(
    (T('Textbook: \\refFHH, Ch.~16 (VaR) and Ch.~17 (copulae); surveys: \\refBFLV; \\refBCHP',
       'Manual: \\refFHH, cap.~16 (VaR) și cap.~17 (copule); sinteze: \\refBFLV; \\refBCHP'),
     [T('the landmark papers of the chapter: \\refAB; \\refAPPR; \\refBE; \\refBGLP; \\refDYc', 'lucrările de referință ale capitolului: \\refAB; \\refAPPR; \\refBE; \\refBGLP; \\refDYc')]),
    T('Python Quantlets of this chapter: \\href{' + QLURL + '}{Quantlets/Ch\\_15}', 'Quantlet-urile Python ale capitolului: \\href{' + QLURL + '}{Quantlets/Ch\\_15}'),
    T('Lecture notebook: \\href{\\colaburl{notebooks/EN/chapter15_lecture_notebook.ipynb}}{open in Google Colab}',
      'Notebook-ul cursului: \\href{\\colaburl{notebooks/EN/chapter15_lecture_notebook.ipynb}}{deschideți în Google Colab}'),
    T('Video courses: \\quantinar{Financial Risk Meter for Emerging Markets}{https://quantinar.com/course/52/FRM}; \\quantinar{Dynamic Crypto Networks}{https://quantinar.com/course/50/cryptonetworks}',
      'Cursuri video: \\quantinar{Financial Risk Meter for Emerging Markets}{https://quantinar.com/course/52/FRM}; \\quantinar{Dynamic Crypto Networks}{https://quantinar.com/course/50/cryptonetworks}')))

# =============================================================================
# 1. CE ESTE RISCUL SISTEMIC
# =============================================================================
D.section('What Is Systemic Risk?', 'Ce este riscul sistemic?')

D.frame(T('A Definition', 'O definiție'), items(
    (T('\\textbf{Systemic risk}: the risk that the financial system as a whole stops working (lending, payments, trading), with large costs for the real economy \\refFSB',
       '\\textbf{Riscul sistemic}: riscul ca sistemul financiar în ansamblu să nu mai funcționeze (creditare, plăți, tranzacționare), cu costuri mari pentru economia reală \\refFSB'),
     [T('the definition of the IMF (International Monetary Fund), the BIS and the FSB (Financial Stability Board) for the G-20 in 2009', 'definiția dată în 2009 pentru G-20 de IMF (International Monetary Fund, Fondul Monetar Internațional), BIS și FSB (Financial Stability Board, Consiliul pentru Stabilitate Financiară)')]),
    (T('Three criteria for a \\textbf{systemically important} institution \\refFSB', 'Trei criterii pentru o instituție \\textbf{importantă sistemic} \\refFSB'),
     [T('\\textbf{size}: how much of the system\'s services it provides', '\\textbf{mărimea}: ce parte din serviciile sistemului oferă'),
      T('\\textbf{substitutability}: whether others can take over its activity quickly', '\\textbf{substituibilitatea}: dacă alte instituții îi pot prelua rapid activitatea'),
      T('\\textbf{interconnectedness}: how many others it would drag down', '\\textbf{interconectarea}: pe cîte alte instituții le-ar trage în jos')]),
    T('Key point: systemic risk is a property of the \\textbf{system}, not of a single balance sheet', 'Ideea principală: riscul sistemic este o proprietate a \\textbf{sistemului}, nu a unui singur bilanț')))

D.frame(T('Why VaR of Each Bank Is Not Enough', 'De ce nu ajunge VaR-ul fiecărei bănci'), items(
    (T('VaR 1\\% of a bank (Chapter 10): the loss of \\textbf{this} bank exceeded with probability 1\\%', 'VaR 1\\% al unei bănci (Capitolul 10): pierderea \\textbf{acestei} bănci, depășită cu probabilitatea 1\\%'),
     [T('it ignores what happens to the others when this bank is in trouble', 'ignoră ce se întîmplă cu celelalte bănci cînd această bancă are probleme')]),
    (T('\\textbf{Fallacy of composition}: if every bank sells risky assets to cut its own VaR, prices fall for all and the risk of all rises',
       '\\textbf{Eroarea de compoziție}: dacă fiecare bancă vinde active riscante ca să-și reducă propriul VaR, prețurile scad pentru toate și riscul tuturor crește'),
     [T('a small bank with a high VaR can matter less than a large, connected bank with a moderate VaR', 'o bancă mică, cu VaR mare, poate conta mai puțin decît o bancă mare și interconectată, cu VaR moderat')]),
    (T('We need measures of the \\textbf{tail of the system given the institution}, or of the \\textbf{institution given the tail of the system}',
       'Avem nevoie de măsuri ale \\textbf{cozii sistemului condiționat de instituție} sau ale \\textbf{instituției condiționat de coada sistemului}'),
     [T('CoVaR: the system given that the bank is in distress \\refAB', 'CoVaR: sistemul, dat fiind că banca este în dificultate \\refAB'),
      T('MES and SRISK: the bank given that the market crashes \\refAPPR; \\refBE', 'MES și SRISK: banca, dat fiind că piața se prăbușește \\refAPPR; \\refBE')])))

D.frame(T('Channel 1: Direct Contagion', 'Canalul 1: contagiunea directă'), items(
    (T('\\textbf{Contagion}: the failure of one institution causes losses or failures at others', '\\textbf{Contagiunea}: falimentul unei instituții produce pierderi sau falimente la altele'),
     [T('direct exposures: interbank loans, derivatives, deposits held at other banks', 'expuneri directe: credite interbancare, instrumente derivate, depozite ținute la alte bănci'),
      T('a network of balance sheets: a loss travels along the links \\refAG', 'o rețea de bilanțuri: o pierdere se propagă de-a lungul legăturilor \\refAG')]),
    (T('Network structure matters \\refAG; \\refHM', 'Structura rețelei contează \\refAG; \\refHM'),
     [T('many links spread small shocks thin, so the system absorbs them', 'multe legături împrăștie șocurile mici, astfel încît sistemul le absoarbe'),
      T('the same links transmit large shocks everywhere: ``robust yet fragile\'\'', 'aceleași legături transmit șocurile mari peste tot: „robust, dar fragil”')]),
    T('\\textbf{Too interconnected to fail}: an institution whose failure would hit many counterparties at once', '\\textbf{Too interconnected to fail} (prea interconectată pentru a fi lăsată să falimenteze): o instituție al cărei faliment ar lovi simultan multe contrapărți')))

D.frame(T('Channel 2: Common Exposures and Fire Sales', 'Canalul 2: expuneri comune și vînzări forțate'), items(
    (T('\\textbf{Common exposures}: banks hold the same assets (mortgages, government bonds), so one shock hits all of them at once',
       '\\textbf{Expunerile comune}: băncile dețin aceleași active (credite ipotecare, obligațiuni de stat), astfel încît un singur șoc le lovește pe toate simultan'),
     [T('no link between the banks is needed: the correlation comes from the assets', 'nu este nevoie de nicio legătură între bănci: corelația vine din active')]),
    (T('\\textbf{Fire sales}: a bank that must sell quickly sells below the fundamental value \\refSV', '\\textbf{Vînzările forțate} (fire sales): o bancă ce trebuie să vîndă repede vinde sub valoarea fundamentală \\refSV'),
     [T('the lower price is a loss for every holder of the asset (marked to market)', 'prețul mai mic este o pierdere pentru toți deținătorii activului (evaluat la prețul pieței)'),
      T('\\textbf{liquidity spiral}: losses $\\to$ higher margins $\\to$ more sales $\\to$ lower prices \\refBP', '\\textbf{spirala lichidității}: pierderi $\\to$ marje mai mari $\\to$ mai multe vînzări $\\to$ prețuri mai mici \\refBP')]),
    T('\\textbf{Leverage} amplifies everything: with assets 20 times equity, a 5\\% fall in asset value wipes out the equity',
      '\\textbf{Efectul de levier} amplifică totul: cu active de 20 de ori mai mari decît capitalul propriu, o scădere de 5\\% a valorii activelor anulează capitalul')))

D.frame(T('Channel 3: Bank Runs and Information Contagion', 'Canalul 3: retragerea depozitelor și contagiunea informațională'), cols(items(
    (T('\\textbf{Bank run}: depositors withdraw at once because they fear that others will withdraw first \\refDD', '\\textbf{Bank run} (retragerea masivă a depozitelor): deponenții își retrag banii simultan, de teamă că alții îi vor retrage primii \\refDD'),
     [T('a bank holds long, illiquid loans and short, liquid deposits: it cannot pay everyone at once', 'o bancă deține credite pe termen lung, nelichide, și depozite pe termen scurt, lichide: nu poate plăti tuturor simultan'),
      T('a run can hit a solvent bank; deposit insurance removes the reason to run', 'o retragere masivă poate lovi și o bancă solvabilă; garantarea depozitelor elimină motivul retragerii')]),
    (T('\\textbf{Information contagion}: trouble at one bank makes investors doubt similar banks', '\\textbf{Contagiunea informațională}: problemele unei bănci îi fac pe investitori să se îndoiască de băncile similare'),
     [T('Northern Rock (United Kingdom), September 2007: the first run on a British bank in over a century', 'Northern Rock (Regatul Unit), septembrie 2007: prima retragere masivă a depozitelor de la o bancă britanică din ultimul secol')])),
    ph('nrock', T('Depositors queue outside a Northern Rock branch, September 2007', 'Deponenți la coadă în fața unei sucursale Northern Rock, septembrie 2007'), h='0.40\\textheight'),
    wl='0.56', wr='0.40'), 'footnotesize')

D.frame(T('Too Big to Fail', 'Too big to fail'), items(
    (T('\\textbf{Too big to fail}: a bank whose failure would damage the economy so much that the state rescues it', '\\textbf{Too big to fail} (prea mare pentru a fi lăsată să falimenteze): o bancă al cărei faliment ar afecta economia atît de mult încît statul o salvează'),
     [T('the expected rescue lowers its funding cost: an implicit subsidy', 'salvarea așteptată îi reduce costul de finanțare: o subvenție implicită'),
      T('moral hazard: the bank takes more risk because losses are shared with taxpayers', 'hazard moral: banca își asumă mai mult risc, deoarece pierderile sînt împărțite cu contribuabilii')]),
    (T('The regulatory answer after 2008 \\refGSIB', 'Răspunsul autorităților după 2008 \\refGSIB'),
     [T('G-SIBs (global systemically important banks): a list published every year, with extra capital for each bank', 'G-SIB (global systemically important banks, bănci de importanță sistemică globală): o listă publicată anual, cu capital suplimentar pentru fiecare bancă'),
      T('resolution plans: a failing bank must be wound down without taxpayers\' money', 'planuri de rezoluție: o bancă aflată în dificultate trebuie lichidată fără banii contribuabililor')]),
    T('Market measures (CoVaR, SRISK) let us rank banks without waiting for a failure', 'Măsurile din datele de piață (CoVaR, SRISK) permit ordonarea băncilor fără a aștepta un faliment')))

TB = '>{\\raggedright\\arraybackslash}'
D.frame(T('Measures of This Chapter at a Glance', 'Măsurile capitolului, pe scurt'), table(
    TB + 'p{2.4cm}' + TB + 'p{5.6cm}' + TB + 'p{3.4cm}',
    T('\\textbf{Measure}', '\\textbf{Măsura}') + ' & ' + T('\\textbf{Question}', '\\textbf{Întrebarea}') + ' & ' + T('\\textbf{Tool}', '\\textbf{Instrumentul}'),
    [T('correlation, tail dependence', 'corelația, dependența în cozi') + ' & ' + T('do banks and markets crash together?', 'se prăbușesc băncile și piețele împreună?') + ' & ' + T('pseudo-observations, t copula', 'pseudo-observații, copula t'),
     'CoVaR, $\\Delta$CoVaR & ' + T('how much worse is the system when the bank is in distress?', 'cu cît este mai rău sistemul cînd banca este în dificultate?') + ' & ' + T('quantile regression', 'regresia cuantilică'),
     'MES & ' + T('how much does the bank lose on the market\'s worst days?', 'cît pierde banca în cele mai proaste zile ale pieței?') + ' & ' + T('conditional average', 'medie condiționată'),
     'SRISK & ' + T('how much capital would the bank miss in a crisis?', 'cît capital i-ar lipsi băncii într-o criză?') + ' & ' + T('MES, leverage', 'MES, efectul de levier'),
     T('networks', 'rețele') + ' & ' + T('who transmits shocks to whom?', 'cine transmite șocuri și cui?') + ' & ' + T('Granger tests, variance decompositions', 'teste Granger, descompunerea varianței')],
    size='footnotesize') + items(
    T('All measures use only market prices: they are available daily, for any listed bank', 'Toate măsurile folosesc doar prețurile de piață: sînt disponibile zilnic, pentru orice bancă listată'),
    T('Their weakness: market prices can be calm while risk builds up (2006--2007)', 'Punctul lor slab: prețurile de piață pot fi liniștite în timp ce riscul se acumulează (2006--2007)')))

D.recap(('What Is Systemic Risk?', 'ce este riscul sistemic?'), [
    T('Systemic risk: the system stops working; it is a property of the system, not of one balance sheet', 'Riscul sistemic: sistemul nu mai funcționează; este o proprietate a sistemului, nu a unui singur bilanț'),
    T('Channels: direct contagion, common exposures and fire sales, runs and information contagion, all amplified by leverage', 'Canale: contagiunea directă, expunerile comune și vînzările forțate, retragerile masive și contagiunea informațională, toate amplificate de efectul de levier'),
    T('Measures look at joint tails: the system given the bank (CoVaR) or the bank given the system (MES, SRISK)', 'Măsurile privesc cozile comune: sistemul condiționat de bancă (CoVaR) sau banca condiționat de sistem (MES, SRISK)')])

# =============================================================================
# 2. PATRU CRIZE
# =============================================================================
D.section('Four Crises in the Data', 'Patru crize în date')

D.frame(T('Data of the Chapter', 'Datele capitolului'), table(
    'lll', T('\\textbf{Group}', '\\textbf{Grupul}') + ' & ' + T('\\textbf{Banks}', '\\textbf{Băncile}') + ' & ' + T('\\textbf{Market index}', '\\textbf{Indicele de piață}'),
    [T('US banks', 'Bănci americane') + ' & JPMorgan Chase, Bank of America, Citigroup, Goldman Sachs, Morgan Stanley, Wells Fargo & S\\&P 500',
     T('European banks', 'Bănci europene') + ' & Deutsche Bank, BNP Paribas, Santander, ING, HSBC & Euro Stoxx 50',
     T('Romanian banks', 'Bănci românești') + ' & ' + T('Banca Transilvania (TLV), BRD (since 2010)', 'Banca Transilvania (TLV), BRD (din 2010)') + ' & BET'],
    size='scriptsize') + items(
    T('Daily adjusted closes of the banks and closes of the indices from EODHD, January 2005 -- 18 September 2026; the VIX as a stress indicator',
      'Prețuri de închidere ajustate zilnice ale băncilor și închiderile indicilor, de la EODHD, ianuarie 2005 -- 18 septembrie 2026; VIX ca indicator de stres'),
    T('Prices are joined on common days first, then daily log returns in \\%: $r_t = 100(\\ln P_t - \\ln P_{t-1})$', 'Prețurile se aliniază întîi pe zilele comune, apoi se calculează randamentele logaritmice zilnice în \\%: $r_t = 100(\\ln P_t - \\ln P_{t-1})$'),
    T('BVB banks from 2010: before, many days have no trades; TLV without 30--31 May 2016 (a price adjustment applied one day late, Chapter 2)',
      'Băncile BVB din 2010: înainte, multe zile nu au tranzacții; TLV fără 30--31 mai 2016 (o ajustare de preț aplicată cu o zi mai tîrziu, Capitolul 2)'),
    T('Bank portfolio of a group: equal weights, rebalanced daily', 'Portofoliul de bănci al unui grup: ponderi egale, reechilibrate zilnic')))

chart(T('Bank Portfolios since 2005', 'Portofoliile de bănci din 2005'), 'sfm_ch15_bank_indices', 'SFM_ch15_crises', [
    T('Value of 100 invested in each equal-weighted portfolio (log scale); dotted lines: Lehman, Draghi\'s speech, COVID-19, the failure of SVB',
      'Valoarea a 100 de unități investite în fiecare portofoliu cu ponderi egale (scară logaritmică); linii punctate: Lehman, discursul lui Draghi, COVID-19, falimentul SVB'),
    T('Maximum drawdown: US banks $-@{ix.US.dd}\\%$ (@{ix.US.ddd}), European banks $-@{ix.EU.dd}\\%$ (@{ix.EU.ddd}), Romanian banks $-@{ix.RO.dd}\\%$ (@{ix.RO.ddd})',
      'Drawdown maxim: băncile americane $-@{ix.US.dd}\\%$ (@{ix.US.ddd}), băncile europene $-@{ix.EU.dd}\\%$ (@{ix.EU.ddd}), băncile românești $-@{ix.RO.dd}\\%$ (@{ix.RO.ddd})')])

D.frame(T('Interpretation of the Bank Portfolios', 'Interpretarea portofoliilor de bănci'), items(
    (T('2008--2009: US and European banks lost about four fifths of their value; the trough came in March 2009, six months after Lehman',
       '2008--2009: băncile americane și europene au pierdut aproximativ patru cincimi din valoare; minimul a venit în martie 2009, la șase luni după Lehman'),
     [T('a systemic crisis is a process that lasts months, not a single bad day', 'o criză sistemică este un proces care durează luni, nu o singură zi proastă')]),
    T('European banks stayed below their 2005 level for most of the period: the euro-area crisis and low interest rates', 'Băncile europene au rămas sub nivelul din 2005 în cea mai mare parte a perioadei: criza din zona euro și dobînzile scăzute'),
    T('Romanian banks: their worst fall came in the euro-area crisis of 2011--2012 (the parent banks of many Romanian banks are in the euro area)',
      'Băncile românești: cea mai mare scădere a venit în criza din zona euro din 2011--2012 (băncile-mamă ale multor bănci din România sînt în zona euro)'),
    T('March 2020 and March 2023: sharp falls, but short; central banks and governments reacted within days', 'Martie 2020 și martie 2023: scăderi abrupte, dar scurte; băncile centrale și guvernele au reacționat în cîteva zile')))

D.frame(T('2008: Lehman Brothers', '2008: Lehman Brothers'), cols(items(
    (T('15 September 2008: Lehman Brothers files for bankruptcy, the largest bankruptcy in US history (assets above 600 billion USD)',
       '15 septembrie 2008: Lehman Brothers intră în faliment, cel mai mare faliment din istoria SUA (active de peste 600 de miliarde USD)'),
     [T('16 September: AIG, an insurer that had sold protection on mortgage debt to many banks, is rescued by the Federal Reserve', '16 septembrie: AIG, un asigurător care vînduse multor bănci protecție pentru datoriile ipotecare, este salvat de Rezerva Federală'),
      T('the same day, a money market fund holding Lehman debt ``breaks the buck\'\' and investors run from such funds', 'în aceeași zi, un fond monetar cu titluri Lehman scade sub un dolar pe unitate, iar investitorii își retrag banii din astfel de fonduri')]),
    (T('All three channels at once', 'Toate cele trei canale simultan'),
     [T('direct exposures (derivatives with Lehman), common exposures (mortgages), a run on short-term funding', 'expuneri directe (derivate cu Lehman), expuneri comune (credite ipotecare), retragerea finanțării pe termen scurt')]),
    T('From the end of August to the trough in November: US banks $-@{ep.leh.US.min}\\%$, European banks $-@{ep.leh.EU.min}\\%$, S\\&P 500 $-@{ep.leh.sp500.min}\\%$',
      'De la sfîrșitul lui august pînă la minimul din noiembrie: băncile americane $-@{ep.leh.US.min}\\%$, cele europene $-@{ep.leh.EU.min}\\%$, S\\&P 500 $-@{ep.leh.sp500.min}\\%$')),
    ph('lehman', T('Lehman Brothers headquarters, New York, 15 September 2008', 'Sediul Lehman Brothers, New York, 15 septembrie 2008'), h='0.40\\textheight'),
    wl='0.58', wr='0.38'), 'footnotesize')

D.frame(T('2010--2012: The Euro-Area Crisis', '2010--2012: criza din zona euro'), cols(items(
    (T('Greece (2010), then Ireland, Portugal, Spain and Italy: doubts about government debt', 'Grecia (2010), apoi Irlanda, Portugalia, Spania și Italia: îndoieli privind datoria publică'),
     [T('\\textbf{the bank--sovereign loop}: banks hold their own government\'s bonds, and the government stands behind its banks', '\\textbf{bucla bănci--stat}: băncile dețin obligațiunile propriului stat, iar statul garantează băncile'),
      T('a fall in bond prices weakens the banks; rescuing the banks weakens the state', 'o scădere a prețului obligațiunilor slăbește băncile; salvarea băncilor slăbește statul')]),
    (T('26 July 2012: Mario Draghi, President of the ECB (European Central Bank): ``the ECB is ready to do whatever it takes to preserve the euro\'\' \\refDraghi',
       '26 iulie 2012: Mario Draghi, președintele ECB (European Central Bank, Banca Centrală Europeană): „Banca Centrală Europeană este pregătită să facă tot ce este necesar pentru a păstra moneda euro” \\refDraghi'),
     [T('a promise, not a purchase, stopped the spiral: the role of expectations', 'o promisiune, nu o cumpărare, a oprit spirala: rolul așteptărilor')]),
    T('July 2011 -- September 2012, worst point: European banks $-@{ep.euro.EU.min}\\%$, US banks $-@{ep.euro.US.min}\\%$, Romanian banks $-@{ep.euro.RO.min}\\%$',
      'Iulie 2011 -- septembrie 2012, punctul cel mai slab: băncile europene $-@{ep.euro.EU.min}\\%$, cele americane $-@{ep.euro.US.min}\\%$, cele românești $-@{ep.euro.RO.min}\\%$')),
    ph('ecb', T('The ECB building in Frankfurt am Main', 'Clădirea ECB din Frankfurt am Main'), h='0.30\\textheight'),
    wl='0.60', wr='0.36'), 'footnotesize')

D.frame(T('March 2020: The Dash for Cash', 'Martie 2020: goana după lichiditate'), cols(items(
    (T('COVID-19 lockdowns: everyone wanted cash at the same time, even government bonds were sold', 'Restricțiile COVID-19: toată lumea a vrut lichiditate în același timp; s-au vîndut chiar și obligațiunile de stat'),
     [T('the stress started outside the banks: funds, firms, the bond market', 'stresul a pornit din afara băncilor: fonduri, firme, piața obligațiunilor'),
      T('banks were better capitalised than in 2008 (Basel III) and became part of the solution (credit lines)', 'băncile erau mai bine capitalizate decît în 2008 (Basel III) și au devenit parte a soluției (liniile de credit)')]),
    (T('Fast policy response: central banks bought assets and lent in dollars to other central banks', 'Răspuns rapid al autorităților: băncile centrale au cumpărat active și au împrumutat dolari altor bănci centrale'),
     [T('trough on @{ep.c20.US.dmin}: US banks $-@{ep.c20.US.min}\\%$, European banks $-@{ep.c20.EU.min}\\%$, Romanian banks $-@{ep.c20.RO.min}\\%$, S\\&P 500 $-@{ep.c20.sp500.min}\\%$',
        'minimul pe @{ep.c20.US.dmin}: băncile americane $-@{ep.c20.US.min}\\%$, cele europene $-@{ep.c20.EU.min}\\%$, cele românești $-@{ep.c20.RO.min}\\%$, S\\&P 500 $-@{ep.c20.sp500.min}\\%$')])),
    ph('fidi', T('The Financial District of New York, 25 March 2020', 'Districtul financiar din New York, 25 martie 2020'), h='0.38\\textheight'),
    wl='0.58', wr='0.38'), 'footnotesize')

D.frame(T('March 2023: SVB', 'Martie 2023: SVB'), cols(items(
    (T('SVB (Silicon Valley Bank): deposits of technology firms, mostly above the insured limit, invested in long-term bonds', 'SVB (Silicon Valley Bank): depozite ale firmelor de tehnologie, majoritatea peste plafonul garantat, investite în obligațiuni pe termen lung'),
     [T('rising interest rates in 2022 created large unrealised losses on the bonds', 'creșterea dobînzilor din 2022 a produs pierderi mari nerealizate pe obligațiuni'),
      T('9 March 2023: depositors tried to withdraw about 42 billion USD in one day; 10 March: the bank is closed \\refFed; \\refFDIC',
        '9 martie 2023: deponenții au încercat să retragă aproximativ 42 de miliarde USD într-o singură zi; 10 martie: banca este închisă \\refFed; \\refFDIC')]),
    (T('A run in the age of mobile banking: faster than any run before', 'O retragere masivă în epoca aplicațiilor bancare: mai rapidă decît oricare alta înainte'),
     [T('a mid-sized bank became systemic through information contagion: other regional banks fell the next days', 'o bancă de mărime medie a devenit sistemică prin contagiune informațională: alte bănci regionale au scăzut în zilele următoare')])),
    ph('svb', T('Silicon Valley Bank headquarters, Santa Clara, 13 March 2023', 'Sediul Silicon Valley Bank, Santa Clara, 13 martie 2023'), h='0.38\\textheight'),
    wl='0.58', wr='0.38'), 'footnotesize')

D.frame(T('March 2023: Credit Suisse', 'Martie 2023: Credit Suisse'), cols(items(
    (T('Credit Suisse: a G-SIB that had lost clients and money for years (scandals, losses on hedge-fund clients)', 'Credit Suisse: o bancă G-SIB care pierdea de ani de zile clienți și bani (scandaluri, pierderi cu clienți de tip hedge fund)'),
     [T('after SVB, investors and depositors left quickly', 'după SVB, investitorii și deponenții au plecat repede')]),
    (T('19 March 2023: takeover by UBS, approved by the Swiss supervisor FINMA \\refFINMA', '19 martie 2023: preluarea de către UBS, aprobată de autoritatea elvețiană de supraveghere FINMA \\refFINMA'),
     [T('AT1 bonds (additional tier 1 capital) of about 16 billion CHF were written down to zero', 'obligațiunile AT1 (additional tier 1, capital suplimentar de nivel 1) de aproximativ 16 miliarde CHF au fost anulate'),
      T('a rescue by merger, not a resolution: too big to fail was still alive', 'o salvare prin fuziune, nu o rezoluție: problema too big to fail nu dispăruse')]),
    T('From the end of February to the trough: European banks $-@{ep.c23.EU.min}\\%$, US banks $-@{ep.c23.US.min}\\%$, Romanian banks $-@{ep.c23.RO.min}\\%$',
      'De la sfîrșitul lui februarie pînă la minim: băncile europene $-@{ep.c23.EU.min}\\%$, cele americane $-@{ep.c23.US.min}\\%$, cele românești $-@{ep.c23.RO.min}\\%$')),
    ph('cs', T('The Credit Suisse headquarters at Paradeplatz, Zürich', 'Sediul Credit Suisse din Paradeplatz, Zürich'), h='0.28\\textheight'),
    wl='0.60', wr='0.36'), 'footnotesize')

chart(T('Four Episodes Side by Side', 'Patru episoade comparate'), 'sfm_ch15_episodes', 'SFM_ch15_crises', [
    T('Cumulative return from the start of each window; the Romanian banks enter the data in 2010', 'Randamentul cumulat de la începutul fiecărei ferestre; băncile românești intră în date în 2010'),
    T('Interpretation: in 2008 and 2011--2012 the banks fell much more than the S\\&P 500 (the banks were the problem); in 2020 they fell with the market; in 2023 they fell while the market rose',
      'Interpretare: în 2008 și 2011--2012 băncile au scăzut mult mai mult decît S\\&P 500 (băncile erau problema); în 2020 au scăzut odată cu piața; în 2023 au scăzut în timp ce piața creștea')],
    h='0.64\\textheight')

D.recap(('Four Crises', 'patru crize'), [
    T('2008: contagion through derivatives, common mortgage exposures and a run on short-term funding', '2008: contagiune prin derivate, expuneri comune la creditele ipotecare și retragerea finanțării pe termen scurt'),
    T('2011--2012: the bank--sovereign loop; stopped by a central-bank promise', '2011--2012: bucla bănci--stat; oprită de o promisiune a băncii centrale'),
    T('2020: a shock from outside, met by better-capitalised banks; 2023: digital runs and information contagion', '2020: un șoc din exterior, întîmpinat de bănci mai bine capitalizate; 2023: retrageri digitale și contagiune informațională')])

# =============================================================================
# 3. CORELAȚIE ȘI DEPENDENȚĂ ÎN COZI
# =============================================================================
D.section('Correlation and Tail Dependence in Crises', 'Corelația și dependența în cozi în crize')

chart(T('Correlation between Banks Rises in Crises', 'Corelația dintre bănci crește în crize'), 'sfm_ch15_rolling_corr', 'SFM_ch15_dependence', [
    T('Average of the pairwise correlations of daily returns within each group, on 120-day windows', 'Media corelațiilor dintre perechile de bănci, pe randamente zilnice, în fiecare grup, pe ferestre de 120 de zile'),
    T('US banks: $@{co.US.calm}$ before the crisis (July 2005 -- June 2007), up to $@{co.US.max}$ (@{co.US.dmax}); TLV and BRD: $@{co.RO.y2017}$ in 2017, $@{co.RO.c2020}$ in April--June 2020',
      'Băncile americane: $@{co.US.calm}$ înainte de criză (iulie 2005 -- iunie 2007), pînă la $@{co.US.max}$ (@{co.US.dmax}); TLV și BRD: $@{co.RO.y2017}$ în 2017, $@{co.RO.c2020}$ în aprilie--iunie 2020')],
    h='0.62\\textheight')

D.frame(T('Interpretation: When Diversification Fails', 'Interpretarea corelațiilor: cînd diversificarea nu mai funcționează'), items(
    (T('In crises, the banks move together: a portfolio of several banks is almost one bank', 'În crize, băncile se mișcă împreună: un portofoliu de mai multe bănci este aproape o singură bancă'),
     [T('the common factor (the market, funding stress) dominates the bank-specific news', 'factorul comun (piața, stresul de finanțare) domină știrile specifice fiecărei bănci')]),
    (T('The two Romanian banks: low correlation in calm years, very high in 2020', 'Cele două bănci românești: corelație mică în anii liniștiți, foarte mare în 2020'),
     [T('in calm years the BVB stocks are driven by their own news and by low liquidity; in a crisis, by the common shock', 'în anii liniștiți, acțiunile BVB sînt determinate de propriile știri și de lichiditatea redusă; într-o criză, de șocul comun')]),
    T('Caution: a rolling correlation estimated on 120 days is noisy, and a rise in correlation does not prove contagion (next slide)', 'Atenție: o corelație pe 120 de zile este zgomotoasă, iar o creștere a corelației nu dovedește contagiunea (slide-ul următor)')))

D.frame(T('Contagion or Interdependence? Forbes and Rigobon', 'Contagiune sau interdependență? Forbes și Rigobon'), items(
    (T('If $y = \\beta x + e$ with fixed $\\beta$, the correlation of $x$ and $y$ rises \\textbf{automatically} when the variance of $x$ rises \\refFR',
       'Dacă $y = \\beta x + e$ cu $\\beta$ fix, corelația dintre $x$ și $y$ crește \\textbf{automat} cînd crește varianța lui $x$ \\refFR'),
     [T('a higher correlation in a crisis may be the same link seen in a more volatile period: \\textbf{interdependence}', 'o corelație mai mare într-o criză poate fi aceeași legătură, văzută într-o perioadă mai volatilă: \\textbf{interdependență}'),
      T('\\textbf{contagion}: the link itself becomes stronger', '\\textbf{contagiune}: legătura însăși devine mai puternică')]),
    (T('Corrected correlation: $\\rho^* = \\rho / \\sqrt{1 + \\delta(1 - \\rho^2)}$, with $\\delta = \\sigma^2_{\\text{crisis}} / \\sigma^2_{\\text{calm}} - 1$ for the source market',
       'Corelația corectată: $\\rho^* = \\rho / \\sqrt{1 + \\delta(1 - \\rho^2)}$, cu $\\delta = \\sigma^2_{\\text{criză}} / \\sigma^2_{\\text{calm}} - 1$ pentru piața-sursă'),
     [T('US bank portfolio (source) and European bank portfolio: calm January 2005 -- June 2007 ($n = @{fr.n0}$), crisis 15 September -- 31 December 2008 ($n = @{fr.n1}$)',
        'Portofoliul băncilor americane (sursa) și cel al băncilor europene: perioada calmă ianuarie 2005 -- iunie 2007 ($n = @{fr.n0}$), criza 15 septembrie -- 31 decembrie 2008 ($n = @{fr.n1}$)'),
      T('$\\rho_{\\text{calm}} = @{fr.rho_calm}$, $\\rho_{\\text{crisis}} = @{fr.rho_crisis}$; variance $@{fr.vc}$ $\\to$ $@{fr.vk}$, so $\\delta = @{fr.delta}$',
        '$\\rho_{\\text{calm}} = @{fr.rho_calm}$, $\\rho_{\\text{criză}} = @{fr.rho_crisis}$; varianța $@{fr.vc}$ $\\to$ $@{fr.vk}$, deci $\\delta = @{fr.delta}$'),
      T('$\\rho^* = @{fr.rho_crisis}/\\sqrt{1 + @{fr.delta} \\times @{fr.r2}} = @{fr.rho_crisis}/\\sqrt{@{fr.inside}} = @{fr.rho_adj}$',
        '$\\rho^* = @{fr.rho_crisis}/\\sqrt{1 + @{fr.delta} \\times @{fr.r2}} = @{fr.rho_crisis}/\\sqrt{@{fr.inside}} = @{fr.rho_adj}$')]),
    T('Interpretation: after the correction there is no rise in correlation: the data are consistent with interdependence, not with a stronger link',
      'Interpretare: după corecție, corelația nu mai crește: datele sînt compatibile cu interdependența, nu cu o legătură mai puternică')), 'footnotesize')

D.frame(T('Tail Dependence (Chapter 10)', 'Dependența în cozi (Capitolul 10)'), items(
    (T('Pseudo-observations $U = $ rank$/(n + 1)$; \\textbf{lower tail dependence} at level $q$: $\\lambda(q) = P(U_1 \\le q \\mid U_2 \\le q)$',
       'Pseudo-observațiile $U = $ rang$/(n + 1)$; \\textbf{dependența în coada inferioară} la nivelul $q$: $\\lambda(q) = P(U_1 \\le q \\mid U_2 \\le q)$'),
     [T('under independence $\\lambda(q) = q$; the limit $\\lambda_L = \\lim_{q \\to 0}\\lambda(q)$ is the tail-dependence coefficient', 'la independență, $\\lambda(q) = q$; limita $\\lambda_L = \\lim_{q \\to 0}\\lambda(q)$ este coeficientul de dependență în coadă')]),
    (T('The t copula (Chapter 10) has $\\lambda_L = 2t_{\\nu+1}(-\\sqrt{(\\nu + 1)(1 - \\rho)/(1 + \\rho)}) > 0$ \\refDM; the Gaussian copula has $\\lambda_L = 0$',
       'Copula t (Capitolul 10) are $\\lambda_L = 2t_{\\nu+1}(-\\sqrt{(\\nu + 1)(1 - \\rho)/(1 + \\rho)}) > 0$ \\refDM; copula Gaussiană are $\\lambda_L = 0$'),
     [T('$\\rho = \\sin(\\pi\\tau/2)$ from Kendall\'s $\\tau$, $\\nu$ by maximum likelihood', '$\\rho = \\sin(\\pi\\tau/2)$ din $\\tau$ al lui Kendall, $\\nu$ prin verosimilitate maximă')]),
    T('For systemic risk: tail dependence measures how often a bank and the market are in their worst days \\textbf{together}', 'Pentru riscul sistemic: dependența în cozi măsoară cît de des sînt o bancă și piața \\textbf{împreună} în cele mai proaste zile ale lor')))

D.frame(T('Tail Dependence of Banks and Markets', 'Dependența în cozi dintre bănci și piețe'), table(
    'lrrrrrr', T('\\textbf{Pair}', '\\textbf{Perechea}') + ' & $n$ & ' + T('corr.', 'corel.') + ' & $\\lambda(5\\%)$ & $\\lambda(1\\%)$ & $\\hat\\nu$ & $\\lambda_L$ (t)',
    [T('US banks, S\\&P 500', 'bănci americane, S\\&P 500') + ' & @{td.US.n} & $@{td.US.corr}$ & $@{td.US.td5}$ & $@{td.US.td1}$ & $@{td.US.nu}$ & $@{td.US.lambda}$',
     T('European banks, Euro Stoxx 50', 'bănci europene, Euro Stoxx 50') + ' & @{td.EU.n} & $@{td.EU.corr}$ & $@{td.EU.td5}$ & $@{td.EU.td1}$ & $@{td.EU.nu}$ & $@{td.EU.lambda}$',
     T('Romanian banks, BET', 'bănci românești, BET') + ' & @{td.RO.n} & $@{td.RO.corr}$ & $@{td.RO.td5}$ & $@{td.RO.td1}$ & $@{td.RO.nu}$ & $@{td.RO.lambda}$',
     T('European banks, US banks', 'bănci europene, bănci americane') + ' & @{td.USEU.n} & $@{td.USEU.corr}$ & $@{td.USEU.td5}$ & $@{td.USEU.td1}$ & $@{td.USEU.nu}$ & $@{td.USEU.lambda}$'],
    size='scriptsize') + items(
    T('$\\lambda(q)$: empirical; $\\hat\\nu$ and $\\lambda_L$: t copula; under independence $\\lambda(5\\%) = 0.05$ and $\\lambda(1\\%) = 0.01$', '$\\lambda(q)$: empirică; $\\hat\\nu$ și $\\lambda_L$: copula t; la independență, $\\lambda(5\\%) = 0{,}05$ și $\\lambda(1\\%) = 0{,}01$'),
    T('Interpretation: on more than half of the market\'s worst days the banks are also in their worst days; the small $\\hat\\nu$ (about 3--4) means strong joint crashes, which a Gaussian copula misses entirely',
      'Interpretare: în peste jumătate dintre cele mai proaste zile ale pieței, și băncile sînt în cele mai proaste zile ale lor; $\\hat\\nu$ mic (circa 3--4) înseamnă prăbușiri simultane puternice, pe care o copulă Gaussiană nu le surprinde deloc'),
    T('$\\lambda(1\\%)$ rests on a few dozen days: an imprecise estimate', '$\\lambda(1\\%)$ se sprijină pe cîteva zeci de zile: o estimare imprecisă')) + ql('SFM_ch15_dependence'), 'footnotesize')

chart(T('Joint Bad Days', 'Zile proaste comune'), 'sfm_ch15_tail_scatter', 'SFM_ch15_dependence', [
    T('Lower-left corner of the pseudo-observations; orange: both the bank portfolio and the index in their worst 5\\% of days',
      'Colțul din stînga jos al pseudo-observațiilor; portocaliu: atît portofoliul de bănci, cît și indicele sînt în cele mai proaste 5\\% dintre zile'),
    T('US: @{tf.US.both} joint bad days against @{tf.US.exp} under independence (about @{tf.US.ratio} times more); Romania: @{tf.RO.both} against @{tf.RO.exp}',
      'SUA: @{tf.US.both} zile proaste comune, față de @{tf.US.exp} la independență (de circa @{tf.US.ratio} ori mai multe); România: @{tf.RO.both}, față de @{tf.RO.exp}')],
    h='0.60\\textheight')

D.recap(('Correlation and Tail Dependence', 'corelația și dependența în cozi'), [
    T('Bank correlations rise in crises; diversification across banks fails when it is needed', 'Corelațiile dintre bănci cresc în crize; diversificarea între bănci nu funcționează tocmai cînd este nevoie de ea'),
    T('A rise in correlation is not proof of contagion: correct for the rise in volatility (Forbes--Rigobon)', 'O creștere a corelației nu este o dovadă de contagiune: corectați pentru creșterea volatilității (Forbes--Rigobon)'),
    T('Tail dependence (t copula, $\\nu$ about 3--4) shows strong joint crashes of banks and markets', 'Dependența în cozi (copula t, $\\nu$ în jur de 3--4) arată prăbușiri simultane puternice ale băncilor și piețelor')])

# =============================================================================
# 4. CoVaR
# =============================================================================
D.section('CoVaR and Delta-CoVaR', 'CoVaR și Delta-CoVaR')

D.frame(T('From VaR to CoVaR', 'De la VaR la CoVaR'), items(
    (T('$X^i$: return of institution $i$; $X^{sys}$: return of the system; level $\\alpha$ = probability of the tail (Chapter 10)', '$X^i$: randamentul instituției $i$; $X^{sys}$: randamentul sistemului; nivelul $\\alpha$ = probabilitatea cozii (Capitolul 10)'),
     [T('$\\mathrm{VaR}^i_\\alpha = -q_\\alpha(X^i)$: the loss of $i$ exceeded with probability $\\alpha$', '$\\mathrm{VaR}^i_\\alpha = -q_\\alpha(X^i)$: pierderea lui $i$ depășită cu probabilitatea $\\alpha$')]),
    (T('\\textbf{CoVaR} (conditional VaR of the system) \\refAB: $\\mathrm{CoVaR}^{sys|i}_\\alpha = -q_\\alpha\\big(X^{sys} \\mid X^i = q_\\alpha(X^i)\\big)$',
       '\\textbf{CoVaR} (VaR-ul condiționat al sistemului) \\refAB: $\\mathrm{CoVaR}^{sys|i}_\\alpha = -q_\\alpha\\big(X^{sys} \\mid X^i = q_\\alpha(X^i)\\big)$'),
     [T('the VaR 1\\% of the system on a day when institution $i$ is at its own VaR 1\\%: ``CoVaR 1\\%\'\'', 'VaR 1\\% al sistemului într-o zi în care instituția $i$ se află la propriul VaR 1\\%: „CoVaR 1\\%”')]),
    (T('\\textbf{$\\Delta$CoVaR}: $\\Delta\\mathrm{CoVaR}^{sys|i}_\\alpha = \\mathrm{CoVaR}^{sys|X^i = q_\\alpha}_\\alpha - \\mathrm{CoVaR}^{sys|X^i = q_{50\\%}}_\\alpha$',
       '\\textbf{$\\Delta$CoVaR}: $\\Delta\\mathrm{CoVaR}^{sys|i}_\\alpha = \\mathrm{CoVaR}^{sys|X^i = q_\\alpha}_\\alpha - \\mathrm{CoVaR}^{sys|X^i = q_{50\\%}}_\\alpha$'),
     [T('how much the VaR of the system rises when $i$ moves from a normal day (its median) to distress: the \\textbf{contribution} of $i$ to systemic risk',
        'cu cît crește VaR-ul sistemului cînd $i$ trece de la o zi obișnuită (mediana sa) la o zi de criză: \\textbf{contribuția} lui $i$ la riscul sistemic')])))

D.frame(T('Quantile Regression (Chapter 13)', 'Regresia cuantilică (Capitolul 13)'), items(
    (T('Linear quantile regression \\refKB: $q_\\alpha(Y \\mid X = x) = a_\\alpha + b_\\alpha x$', 'Regresia cuantilică liniară \\refKB: $q_\\alpha(Y \\mid X = x) = a_\\alpha + b_\\alpha x$'),
     [T('estimated by minimising $\\sum_t \\rho_\\alpha(y_t - a - bx_t)$, with the check loss $\\rho_\\alpha(u) = u(\\alpha - \\mathbf{1}\\{u < 0\\})$', 'estimată prin minimizarea $\\sum_t \\rho_\\alpha(y_t - a - bx_t)$, cu funcția de pierdere $\\rho_\\alpha(u) = u(\\alpha - \\mathbf{1}\\{u < 0\\})$'),
      T('for $\\alpha = 1\\%$, a residual below the line costs 99 times more than one above it: the line passes below 99\\% of the points', 'pentru $\\alpha = 1\\%$, un reziduu sub dreaptă costă de 99 de ori mai mult decît unul deasupra ei: dreapta trece sub 99\\% dintre puncte')]),
    (T('Same idea as VaR 1\\% by quantile regression in Chapter 13, now with another institution as the regressor', 'Aceeași idee ca VaR 1\\% prin regresie cuantilică din Capitolul 13, acum cu o altă instituție ca variabilă explicativă'),
     [T('the slope $b_\\alpha$ can differ from the OLS slope: the tail can react more strongly than the centre', 'panta $b_\\alpha$ poate diferi de panta OLS: coada poate reacționa mai puternic decît centrul')]),
    T('$a_\\alpha$ is the intercept (constant term) of the regression, $b_\\alpha$ the slope', '$a_\\alpha$ este termenul liber al regresiei, iar $b_\\alpha$ este panta')))

D.frame(T('Estimating CoVaR in Three Steps', 'Estimarea CoVaR în trei pași'), items(
    (T('\\textbf{Step 1}: quantile regression of the system on the institution at level $\\alpha$: $X^{sys}_t = a + bX^i_t + \\varepsilon_t$ \\refAB',
       '\\textbf{Pasul 1}: regresia cuantilică a sistemului pe instituție, la nivelul $\\alpha$: $X^{sys}_t = a + bX^i_t + \\varepsilon_t$ \\refAB'),
     [T('the $\\alpha$-quantile of the error is zero by construction', 'cuantila de ordin $\\alpha$ a erorii este zero prin construcție')]),
    T('\\textbf{Step 2}: the empirical quantiles of the institution: $\\hat q_\\alpha(X^i)$ (distress) and $\\hat q_{50\\%}(X^i)$ (median)', '\\textbf{Pasul 2}: cuantilele empirice ale instituției: $\\hat q_\\alpha(X^i)$ (dificultate) și $\\hat q_{50\\%}(X^i)$ (mediana)'),
    (T('\\textbf{Step 3}: plug in', '\\textbf{Pasul 3}: înlocuim'),
     [T('$\\mathrm{CoVaR}_\\alpha = -(\\hat a + \\hat b\\,\\hat q_\\alpha(X^i))$', '$\\mathrm{CoVaR}_\\alpha = -(\\hat a + \\hat b\\,\\hat q_\\alpha(X^i))$'),
      T('$\\Delta\\mathrm{CoVaR}_\\alpha = \\hat b\\,\\big(\\hat q_{50\\%}(X^i) - \\hat q_\\alpha(X^i)\\big)$: only the slope and the distance between the two quantiles matter',
        '$\\Delta\\mathrm{CoVaR}_\\alpha = \\hat b\\,\\big(\\hat q_{50\\%}(X^i) - \\hat q_\\alpha(X^i)\\big)$: contează doar panta și distanța dintre cele două cuantile')]),
    T('Here the system is proxied by the regional equity index (S\\&P 500, Euro Stoxx 50, BET); Adrian and Brunnermeier use the portfolio of all financial institutions',
      'Aici sistemul este aproximat prin indicele bursier al regiunii (S\\&P 500, Euro Stoxx 50, BET); Adrian și Brunnermeier folosesc portofoliul tuturor instituțiilor financiare')))

D.frame(T('Worked Example: JPMorgan Chase and the S\\&P 500', 'Exemplu rezolvat: JPMorgan Chase și S\\&P 500'), items(
    (T('Daily log returns, 2005--2026 ($n = @{qr.n}$); quantile regression at 1\\%: $\\hat a = @{qr.a}$, $\\hat b = @{qr.b}$',
       'Randamente logaritmice zilnice, 2005--2026 ($n = @{qr.n}$); regresia cuantilică la 1\\%: $\\hat a = @{qr.a}$, $\\hat b = @{qr.b}$'),
     [T('JPM: $\\hat q_{1\\%} = @{qr.q_a}\\%$ (so VaR 1\\% of JPM $= @{qr.var_i}\\%$), median $\\hat q_{50\\%} = @{qr.q_m}\\%$', 'JPM: $\\hat q_{1\\%} = @{qr.q_a}\\%$ (deci VaR 1\\% al JPM $= @{qr.var_i}\\%$), mediana $\\hat q_{50\\%} = @{qr.q_m}\\%$')]),
    (T('\\textbf{CoVaR 1\\%} $= -(@{qr.a} + @{qr.b} \\times (@{qr.q_a})) = @{qr.ma} + @{qr.bqm} = @{qr.covar}\\%$', '\\textbf{CoVaR 1\\%} $= -(@{qr.a} + @{qr.b} \\times (@{qr.q_a})) = @{qr.ma} + @{qr.bqm} = @{qr.covar}\\%$'),
     [T('CoVaR at the median of JPM: $@{qr.covar_med}\\%$', 'CoVaR la mediana JPM: $@{qr.covar_med}\\%$')]),
    (T('\\textbf{$\\Delta$CoVaR 1\\%} $= @{qr.b} \\times (@{qr.q_m} - (@{qr.q_a})) = @{qr.b} \\times @{qr.diff} = @{qr.dcovar}\\%$', '\\textbf{$\\Delta$CoVaR 1\\%} $= @{qr.b} \\times (@{qr.q_m} - (@{qr.q_a})) = @{qr.b} \\times @{qr.diff} = @{qr.dcovar}\\%$'),
     [T('the unconditional VaR 1\\% of the S\\&P 500 is $@{qr.var_sys}\\%$', 'VaR 1\\% necondiționat al S\\&P 500 este $@{qr.var_sys}\\%$')]),
    T('Interpretation: when JPM is in distress, the loss of the index exceeded with probability 1\\% is @{qr.covar}\\%, against @{qr.covar_med}\\% on a normal JPM day',
      'Interpretare: cînd JPM este în dificultate, pierderea indicelui depășită cu probabilitatea 1\\% este de @{qr.covar}\\%, față de @{qr.covar_med}\\% într-o zi obișnuită pentru JPM')))

chart(T('CoVaR in a Picture', 'CoVaR într-un grafic'), 'sfm_ch15_covar_qr', 'SFM_ch15_covar', [
    T('Red line: the 1\\% quantile of the S\\&P 500 given JPM; green line: the median; dashed: the 1\\% quantile and the median of JPM',
      'Linia roșie: cuantila de 1\\% a S\\&P 500 condiționat de JPM; linia verde: mediana; linii întrerupte: cuantila de 1\\% și mediana JPM'),
    T('$\\Delta$CoVaR is the vertical distance between the red dot and the green square on the red line: $@{qr.dcovar}\\%$', '$\\Delta$CoVaR este distanța pe verticală dintre punctul roșu și pătratul verde de pe dreapta roșie: $@{qr.dcovar}\\%$')],
    h='0.60\\textheight')

chart(T('$\\Delta$CoVaR 1\\% of 13 Banks', '$\\Delta$CoVaR 1\\% pentru 13 bănci'), 'sfm_ch15_dcovar', 'SFM_ch15_covar', [
    T('System: the regional index; 95\\% intervals from a moving-block bootstrap (blocks of 20 days, 200 resamples)', 'Sistemul: indicele regiunii; intervale de 95\\% din bootstrap pe blocuri mobile (blocuri de 20 de zile, 200 de reeșantionări)'),
    T('US banks: @{rg.dus.lo}--@{rg.dus.hi}\\%; European banks: @{rg.deu.lo}--@{rg.deu.hi}\\%; TLV $@{cv.TLV.d}\\%$, BRD $@{cv.BRD.d}\\%$; largest: @{top.dcov}', 'Băncile americane: @{rg.dus.lo}--@{rg.dus.hi}\\%; băncile europene: @{rg.deu.lo}--@{rg.deu.hi}\\%; TLV $@{cv.TLV.d}\\%$, BRD $@{cv.BRD.d}\\%$; cel mai mare: @{top.dcov}')],
    h='0.62\\textheight')

D.frame(T('Interpretation of $\\Delta$CoVaR', 'Interpretarea $\\Delta$CoVaR'), items(
    (T('The intervals overlap: within a region, the data cannot rank the banks precisely', 'Intervalele se suprapun: în interiorul unei regiuni, datele nu pot ordona precis băncile'),
     [T('a ranking of systemic importance needs its uncertainty, like any estimate (bootstrap, Chapter 2)', 'orice ordonare după importanța sistemică trebuie însoțită de incertitudinea ei, ca orice estimare (bootstrap, Capitolul 2)')]),
    (T('European and Romanian banks have a larger $\\Delta$CoVaR than US banks', 'Băncile europene și cele românești au un $\\Delta$CoVaR mai mare decît băncile americane'),
     [T('banks weigh more in the Euro Stoxx 50 and in the BET than in the S\\&P 500: the index is closer to a banking system', 'băncile au o pondere mai mare în Euro Stoxx 50 și în BET decît în S\\&P 500: indicele este mai aproape de un sistem bancar'),
      T('TLV and BRD are among the largest stocks of the BET, so the BET partly measures themselves', 'TLV și BRD sînt printre cele mai mari acțiuni din BET, deci BET le măsoară parțial chiar pe ele')]),
    T('Robustness: the ranking at 5\\% is close to the ranking at 1\\% (Spearman correlation $@{sp.15}$)', 'Robustețe: ordinea la 5\\% este apropiată de cea la 1\\% (corelația Spearman $@{sp.15}$)')))

D.frame(T('Worked Example: Banca Transilvania and the BET', 'Exemplu rezolvat: Banca Transilvania și BET'), items(
    (T('TLV and the BET, daily log returns since 2010 ($n = @{cvn.TLV}$); quantile regression at 1\\%: $\\hat b = @{cv.TLV.b}$', 'TLV și BET, randamente logaritmice zilnice din 2010 ($n = @{cvn.TLV}$); regresia cuantilică la 1\\%: $\\hat b = @{cv.TLV.b}$'),
     [T('TLV: $\\hat q_{1\\%} = -@{cv.TLV.var}\\%$ (VaR 1\\% of TLV), median $@{cvm.TLV}\\%$', 'TLV: $\\hat q_{1\\%} = -@{cv.TLV.var}\\%$ (VaR 1\\% al TLV), mediana $@{cvm.TLV}\\%$')]),
    (T('$\\Delta$CoVaR 1\\% $= @{cv.TLV.b} \\times (@{cvm.TLV} + @{cv.TLV.var}) = @{cv.TLV.d}\\%$; CoVaR 1\\% of the BET $= @{cv.TLV.covar}\\%$', '$\\Delta$CoVaR 1\\% $= @{cv.TLV.b} \\times (@{cvm.TLV} + @{cv.TLV.var}) = @{cv.TLV.d}\\%$; CoVaR 1\\% al BET $= @{cv.TLV.covar}\\%$'),
     [T('95\\% block-bootstrap interval for $\\Delta$CoVaR: [@{cv.TLV.lo}, @{cv.TLV.hi}]', 'intervalul bootstrap pe blocuri de 95\\% pentru $\\Delta$CoVaR: [@{cv.TLV.lo}, @{cv.TLV.hi}]')]),
    (T('Interpretation: a distress day of TLV moves the 1\\% quantile of the BET down by about @{cv.TLV.d} percentage points', 'Interpretare: o zi de criză pentru TLV coboară cuantila de 1\\% a BET cu aproximativ @{cv.TLV.d} puncte procentuale'),
     [T('part of this is mechanical: TLV is one of the largest weights in the BET; a cleaner system would be the BET without TLV', 'o parte este mecanică: TLV are una dintre cele mai mari ponderi în BET; un sistem mai curat ar fi BET fără TLV')])))

chart(T('VaR Is Not $\\Delta$CoVaR', 'VaR nu este $\\Delta$CoVaR'), 'sfm_ch15_var_vs_dcovar', 'SFM_ch15_covar', [
    T('Each point: a bank, its own VaR 1\\% (horizontal) and its $\\Delta$CoVaR 1\\% (vertical); Spearman rank correlation $@{sp.vd}$', 'Fiecare punct: o bancă, propriul VaR 1\\% (orizontal) și $\\Delta$CoVaR 1\\% (vertical); corelația rangurilor Spearman $@{sp.vd}$'),
    T('Interpretation: @{vd.b1} and @{vd.b2} are among the three largest VaR 1\\% and among the three smallest $\\Delta$CoVaR; regulating banks by their own VaR misses systemic risk \\refAB',
      'Interpretare: @{vd.b1} și @{vd.b2} sînt printre primele trei după VaR 1\\% și printre ultimele trei după $\\Delta$CoVaR; reglementarea băncilor după propriul VaR scapă din vedere riscul sistemic \\refAB')],
    h='0.60\\textheight')

D.frame(T('Time-Varying $\\Delta$CoVaR', '$\\Delta$CoVaR variabil în timp'), items(
    (T('Adrian and Brunnermeier let the quantiles depend on lagged \\textbf{state variables} $M_{t-1}$ \\refAB', 'Adrian și Brunnermeier lasă cuantilele să depindă de \\textbf{variabile de stare} $M_{t-1}$ decalate \\refAB'),
     [T('$X^i_t = c + \\gamma^\\top M_{t-1}$ at levels $\\alpha$ and 50\\%: $\\hat q_{\\alpha,t}(X^i)$ and $\\hat q_{50\\%,t}(X^i)$', '$X^i_t = c + \\gamma^\\top M_{t-1}$ la nivelurile $\\alpha$ și 50\\%: $\\hat q_{\\alpha,t}(X^i)$ și $\\hat q_{50\\%,t}(X^i)$'),
      T('$X^{sys}_t = a + bX^i_t + \\delta^\\top M_{t-1}$ at level $\\alpha$; then $\\Delta\\mathrm{CoVaR}_t = \\hat b\\,(\\hat q_{50\\%,t} - \\hat q_{\\alpha,t})$', '$X^{sys}_t = a + bX^i_t + \\delta^\\top M_{t-1}$ la nivelul $\\alpha$; apoi $\\Delta\\mathrm{CoVaR}_t = \\hat b\\,(\\hat q_{50\\%,t} - \\hat q_{\\alpha,t})$')]),
    (T('The paper uses seven state variables (VIX, liquidity and credit spreads, interest rates, market and real-estate returns)', 'Lucrarea folosește șapte variabile de stare (VIX, diferențe de lichiditate și de credit, dobînzi, randamentele pieței și ale sectorului imobiliar)'),
     [T('here, a reduced set: the VIX level and the index return of the previous day', 'aici, un set redus: nivelul VIX și randamentul indicelui din ziua precedentă')]),
    T('The slope $\\hat b$ is fixed; the time variation comes from the distance between the two conditional quantiles of the bank', 'Panta $\\hat b$ este fixă; variația în timp vine din distanța dintre cele două cuantile condiționate ale băncii')))

chart(T('$\\Delta$CoVaR 1\\% through Time', '$\\Delta$CoVaR 1\\% în timp'), 'sfm_ch15_dcovar_dynamic', 'SFM_ch15_covar', [
    T('JPMorgan Chase: average $@{dy.JPM.mean}\\%$, maximum $@{dy.JPM.max}\\%$ (@{dy.JPM.dmax}); Deutsche Bank: maximum $@{dy.DBK.max}\\%$ (@{dy.DBK.dmax}); Banca Transilvania: maximum $@{dy.TLV.max}\\%$ (@{dy.TLV.dmax})',
      'JPMorgan Chase: media $@{dy.JPM.mean}\\%$, maximul $@{dy.JPM.max}\\%$ (@{dy.JPM.dmax}); Deutsche Bank: maximul $@{dy.DBK.max}\\%$ (@{dy.DBK.dmax}); Banca Transilvania: maximul $@{dy.TLV.max}\\%$ (@{dy.TLV.dmax})'),
    T('Interpretation: the contribution of a bank to systemic risk is not a constant; it jumps when the VIX jumps (2008, March 2020) and is small in calm years (JPM in 2017: $@{dy.JPM.y2017}\\%$)',
      'Interpretare: contribuția unei bănci la riscul sistemic nu este o constantă; crește brusc cînd crește VIX (2008, martie 2020) și este mică în anii liniștiți (JPM în 2017: $@{dy.JPM.y2017}\\%$)')],
    h='0.60\\textheight')

D.frame(T('Case Study: Adrian and Brunnermeier (2016)', 'Studiu de caz: Adrian și Brunnermeier (2016)'), cols(items(
    (T('\\refAB: CoVaR of US financial institutions (commercial and investment banks, insurers, brokers), weekly data 1986--2010', '\\refAB: CoVaR pentru instituțiile financiare americane (bănci comerciale și de investiții, asigurători, brokeri), date săptămînale 1986--2010'),
     [T('the system: the portfolio of all financial institutions; seven state variables', 'sistemul: portofoliul tuturor instituțiilor financiare; șapte variabile de stare')]),
    (T('Findings', 'Rezultate'),
     [T('in the cross-section, the VaR of an institution says little about its $\\Delta$CoVaR', 'în secțiunea transversală, VaR-ul unei instituții spune puțin despre $\\Delta$CoVaR-ul ei'),
      T('\\textbf{forward} $\\Delta$CoVaR: size, leverage and maturity mismatch two years earlier predict the contribution in a crisis', '$\\Delta$CoVaR \\textbf{prospectiv}: mărimea, efectul de levier și nepotrivirea scadențelor de acum doi ani prezic contribuția într-o criză'),
      T('a basis for countercyclical capital: charge banks when risk is built up, not when it shows', 'o bază pentru capitalul anticiclic: cerințe mai mari cînd riscul se acumulează, nu cînd devine vizibil')])),
    cols(ph('adrian', 'Tobias Adrian', h='0.32\\textheight'), ph('brunn', 'Markus Brunnermeier', h='0.32\\textheight'), wl='0.48', wr='0.48'),
    wl='0.56', wr='0.42'), 'footnotesize')

D.recap(('CoVaR', 'CoVaR'), [
    T('CoVaR 1\\%: the VaR 1\\% of the system when the institution is at its own VaR 1\\%; $\\Delta$CoVaR: the rise from a normal day to distress', 'CoVaR 1\\%: VaR 1\\% al sistemului cînd instituția este la propriul VaR 1\\%; $\\Delta$CoVaR: creșterea de la o zi obișnuită la una de criză'),
    T('Estimated by quantile regression: $\\Delta$CoVaR $= \\hat b\\,(\\hat q_{50\\%} - \\hat q_\\alpha)$', 'Estimat prin regresie cuantilică: $\\Delta$CoVaR $= \\hat b\\,(\\hat q_{50\\%} - \\hat q_\\alpha)$'),
    T('A bank\'s own VaR and its systemic contribution rank banks differently; $\\Delta$CoVaR varies strongly in time', 'VaR-ul propriu al unei bănci și contribuția ei sistemică ordonează diferit băncile; $\\Delta$CoVaR variază puternic în timp')])

# =============================================================================
# 5. MES ȘI SRISK
# =============================================================================
D.section('MES and SRISK', 'MES și SRISK')

D.frame(T('Marginal Expected Shortfall', 'Marginal expected shortfall (MES)'), items(
    (T('\\textbf{MES} \\refAPPR: the expected loss of institution $i$ on the days when the market is in its tail', '\\textbf{MES} \\refAPPR: pierderea așteptată a instituției $i$ în zilele în care piața este în coada sa'),
     [T('$\\mathrm{MES}^i_\\alpha = -E\\big[X^i \\mid X^m \\le q_\\alpha(X^m)\\big]$; we use the worst 5\\% of market days: ``MES 5\\%\'\'', '$\\mathrm{MES}^i_\\alpha = -E\\big[X^i \\mid X^m \\le q_\\alpha(X^m)\\big]$; folosim cele mai proaste 5\\% dintre zilele pieței: „MES 5\\%”'),
      T('the direction is the opposite of CoVaR: the bank given the market, not the market given the bank', 'direcția este opusă celei din CoVaR: banca condiționat de piață, nu piața condiționat de bancă')]),
    (T('Why ``marginal\'\': ES of the market $= \\sum_i w_i\\,\\mathrm{MES}^i$ when the market is a portfolio with weights $w_i$', 'De ce „marginal”: ES-ul pieței $= \\sum_i w_i\\,\\mathrm{MES}^i$ cînd piața este un portofoliu cu ponderile $w_i$'),
     [T('MES is the contribution of $i$ to the ES of the system (Chapter 10)', 'MES este contribuția lui $i$ la ES-ul sistemului (Capitolul 10)')]),
    T('Estimation: sort the days by the market return, take the worst $\\lceil n\\alpha \\rceil$ days, average the bank\'s returns on those days, change the sign',
      'Estimare: ordonăm zilele după randamentul pieței, luăm cele mai proaste $\\lceil n\\alpha \\rceil$ zile, facem media randamentelor băncii în acele zile și schimbăm semnul')))

D.frame(T('Worked Example: MES from Ten Days', 'Exemplu rezolvat: MES din zece zile'), table(
    'l' + 'r' * 10, T('Day', 'Ziua') + ' & ' + ' & '.join(str(i) for i in range(1, 11)),
    [T('market', 'piața') + ' & ' + ' & '.join(f'${x:+.1f}$'.replace('+', '') for x in TOYM),
     T('bank', 'banca') + ' & ' + ' & '.join(f'${x:+.1f}$'.replace('+', '') for x in TOYB)], size='footnotesize') + items(
    (T('Level $\\alpha = 20\\%$: $\\lceil 10 \\times 0.2 \\rceil = 2$ worst market days: day 5 ($-6.0$) and day 1 ($-4.0$)', 'Nivelul $\\alpha = 20\\%$: $\\lceil 10 \\times 0{,}2 \\rceil = 2$ cele mai proaste zile ale pieței: ziua 5 ($-6{,}0$) și ziua 1 ($-4{,}0$)'),
     [T('MES 20\\% $= -(-9.0 - 6.5)/2 = @{toy.mes}\\%$; ES 20\\% of the market $= -(-6.0 - 4.0)/2 = @{toy.es}\\%$', 'MES 20\\% $= -(-9{,}0 - 6{,}5)/2 = @{toy.mes}\\%$; ES 20\\% al pieței $= -(-6{,}0 - 4{,}0)/2 = @{toy.es}\\%$')]),
    T('Interpretation: on the market\'s bad days this bank loses about 1.5 times as much as the market: it amplifies the crashes',
      'Interpretare: în zilele proaste ale pieței, această bancă pierde de aproximativ 1,5 ori mai mult decît piața: amplifică prăbușirile'),
    T('Note: MES selects the days by the market return, not by the bank return', 'Observație: MES selectează zilele după randamentul pieței, nu după randamentul băncii')))

chart(T('MES 5\\% of 13 Banks', 'MES 5\\% pentru 13 bănci'), 'sfm_ch15_mes', 'SFM_ch15_mes_srisk', [
    T('Average daily loss of each bank on the worst 5\\% of days of its regional index (US and Europe since 2005, Romania since 2010); 95\\% block-bootstrap intervals',
      'Pierderea zilnică medie a fiecărei bănci în cele mai proaste 5\\% dintre zilele indicelui regiunii (SUA și Europa din 2005, România din 2010); intervale bootstrap pe blocuri de 95\\%'),
    T('Largest: Citigroup ($@{me.C.m}\\%$) and ING ($@{me.INGA.m}\\%$); smallest: HSBC ($@{me.HSBA.m}\\%$); TLV $@{me.TLV.m}\\%$, BRD $@{me.BRD.m}\\%$',
      'Cele mai mari: Citigroup ($@{me.C.m}\\%$) și ING ($@{me.INGA.m}\\%$); cel mai mic: HSBC ($@{me.HSBA.m}\\%$); TLV $@{me.TLV.m}\\%$, BRD $@{me.BRD.m}\\%$')],
    h='0.60\\textheight')

D.frame(T('Interpretation: MES and $\\Delta$CoVaR', 'Interpretarea MES și a $\\Delta$CoVaR'), items(
    (T('MES is close to a tail beta: the average market loss on its worst 5\\% of days times the sensitivity of the bank', 'MES este apropiat de un beta al cozii: pierderea medie a pieței în cele mai proaste 5\\% dintre zile, înmulțită cu sensibilitatea băncii'),
     [T('Citigroup: OLS beta $@{me.C.beta}$, MES $@{me.C.m}\\%$; HSBC: beta $@{me.HSBA.beta}$, MES $@{me.HSBA.m}\\%$', 'Citigroup: beta OLS $@{me.C.beta}$, MES $@{me.C.m}\\%$; HSBC: beta $@{me.HSBA.beta}$, MES $@{me.HSBA.m}\\%$')]),
    (T('MES and $\\Delta$CoVaR rank the 13 banks differently: Spearman correlation $@{sp.md}$', 'MES și $\\Delta$CoVaR ordonează diferit cele 13 bănci: corelația Spearman $@{sp.md}$'),
     [T('they answer different questions: ``who suffers in a crash\'\' against ``who adds to the crash\'\'', 'răspund la întrebări diferite: „cine suferă într-o prăbușire” față de „cine contribuie la prăbușire”'),
      T('a regulator should look at several measures, with their intervals \\refBCHP', 'o autoritate de reglementare ar trebui să privească mai multe măsuri, cu intervalele lor \\refBCHP')])))

chart(T('Case Study: Did MES Predict the 2008 Losses?', 'Studiu de caz: a prezis MES pierderile din 2008?'), 'sfm_ch15_mes_crisis', 'SFM_ch15_mes_srisk', [
    T('Design of \\refAPPR: MES 5\\% measured from June 2006 to June 2007, against the return from July 2007 to December 2008; here 11 US and European banks',
      'Designul din \\refAPPR: MES 5\\% măsurat din iunie 2006 pînă în iunie 2007, comparat cu randamentul din iulie 2007 pînă în decembrie 2008; aici 11 bănci americane și europene'),
    T('Spearman correlation $@{mc.sp}$ (p $= @{mc.p}$, $n = @{mc.n}$): the sign of the paper (higher MES, larger loss), but not significant with 11 banks; worst return: @{mc.worst} ($@{mc.C.r}\\%$)',
      'Corelația Spearman $@{mc.sp}$ (p $= @{mc.p}$, $n = @{mc.n}$): semnul din lucrare (MES mai mare, pierdere mai mare), dar nesemnificativ cu 11 bănci; cel mai prost randament: @{mc.worst} ($@{mc.C.r}\\%$)')],
    h='0.58\\textheight')

D.frame(T('From MES to LRMES', 'De la MES la LRMES'), items(
    (T('A crisis lasts months: we need the loss of the bank if the market falls 40\\% over six months: \\textbf{LRMES} (long-run MES)',
       'O criză durează luni: avem nevoie de pierderea băncii dacă piața scade cu 40\\% în șase luni: \\textbf{LRMES} (long-run MES, MES pe termen lung)'),
     [T('approximation of \\refAER: $\\mathrm{LRMES} \\approx 1 - \\exp(-18 \\times \\mathrm{MES})$, with MES the daily loss (as a fraction) on the days when the market falls by more than 2\\%',
        'aproximarea din \\refAER: $\\mathrm{LRMES} \\approx 1 - \\exp(-18 \\times \\mathrm{MES})$, cu MES pierderea zilnică (ca fracție) în zilele în care piața scade cu peste 2\\%')]),
    (T('Example: MES $= 3\\% = 0.03$: $\\mathrm{LRMES} = 1 - e^{-0.54} = 1 - @{ex.exp} = @{ex.lr}\\%$', 'Exemplu: MES $= 3\\% = 0{,}03$: $\\mathrm{LRMES} = 1 - e^{-0{,}54} = 1 - @{ex.exp} = @{ex.lr}\\%$'),
     [T('the bank would lose about @{ex.lr}\\% of its market value in such a crisis', 'banca ar pierde aproximativ @{ex.lr}\\% din valoarea de piață într-o astfel de criză')]),
    T('Our 13 banks: LRMES between @{rg.lr.lo}\\% and @{rg.lr.hi}\\%; Brownlees and Engle estimate it instead from a dynamic model (GARCH and dynamic correlation) \\refBE',
      'Cele 13 bănci: LRMES între @{rg.lr.lo}\\% și @{rg.lr.hi}\\%; Brownlees și Engle îl estimează în schimb dintr-un model dinamic (GARCH și corelație dinamică) \\refBE')))

D.frame(T('SRISK: the Capital Shortfall in a Crisis', 'SRISK: deficitul de capital într-o criză'), items(
    (T('A bank needs equity of at least a fraction $k$ of its assets; $k = 8\\%$ in \\refBE', 'O bancă are nevoie de capital propriu de cel puțin o fracție $k$ din active; $k = 8\\%$ în \\refBE'),
     [T('$D$: book value of debt; $W$: market value of equity; assets $\\approx D + W$', '$D$: valoarea contabilă a datoriilor; $W$: valoarea de piață a capitalului propriu; activele $\\approx D + W$')]),
    (T('\\textbf{SRISK} $= k(D + W_{\\text{crisis}}) - W_{\\text{crisis}}$ with $W_{\\text{crisis}} = (1 - \\mathrm{LRMES})W$:', '\\textbf{SRISK} $= k(D + W_{\\text{criză}}) - W_{\\text{criză}}$ cu $W_{\\text{criză}} = (1 - \\mathrm{LRMES})W$:'),
     [T('$\\mathrm{SRISK} = kD - (1 - k)(1 - \\mathrm{LRMES})\\,W$ \\refBE', '$\\mathrm{SRISK} = kD - (1 - k)(1 - \\mathrm{LRMES})\\,W$ \\refBE'),
      T('SRISK $> 0$: capital missing in a crisis; the \\textbf{aggregate SRISK} (sum of the positive values) is what the state might have to cover', 'SRISK $> 0$: capital lipsă într-o criză; \\textbf{SRISK agregat} (suma valorilor pozitive) este ceea ce statul ar putea fi nevoit să acopere')]),
    (T('Example: $D = 900$, $W = 100$ (billion), LRMES $= @{ex.lr}\\%$', 'Exemplu: $D = 900$, $W = 100$ (miliarde), LRMES $= @{ex.lr}\\%$'),
     [T('SRISK $= 0.08 \\times 900 - 0.92 \\times (1 - @{ex.lr}\\%) \\times 100 = @{ex.kd} - @{ex.cap} = @{ex.srisk}$ billion', 'SRISK $= 0{,}08 \\times 900 - 0{,}92 \\times (1 - @{ex.lr}\\%) \\times 100 = @{ex.kd} - @{ex.cap} = @{ex.srisk}$ miliarde')])))

chart(T('SRISK and Leverage', 'SRISK și efectul de levier'), 'sfm_ch15_srisk', 'SFM_ch15_mes_srisk', [
    T('SRISK$/W = k(L - 1) - (1 - k)(1 - \\mathrm{LRMES})$ with leverage $L = (D + W)/W$; SRISK $> 0$ above $L^* = 1 + (1 - k)(1 - \\mathrm{LRMES})/k$',
      'SRISK$/W = k(L - 1) - (1 - k)(1 - \\mathrm{LRMES})$, cu efectul de levier $L = (D + W)/W$; SRISK $> 0$ peste $L^* = 1 + (1 - k)(1 - \\mathrm{LRMES})/k$'),
    T('Interpretation: @{sr.hi} needs capital in a crisis above a leverage of @{sr.hiL}, @{sr.lo} only above @{sr.loL}; banks typically have $L$ between 10 and 20, so leverage decides the sign',
      'Interpretare: @{sr.hi} are nevoie de capital într-o criză peste un efect de levier de @{sr.hiL}, @{sr.lo} doar peste @{sr.loL}; băncile au de regulă $L$ între 10 și 20, deci efectul de levier decide semnul')],
    h='0.58\\textheight')

D.frame(T('Case Study: SRISK and V-Lab', 'Studiu de caz: SRISK și V-Lab'), cols(items(
    (T('\\refBE: SRISK of large US financial firms, with LRMES from a GARCH--DCC model (Chapter 9)', '\\refBE: SRISK pentru marile firme financiare americane, cu LRMES dintr-un model GARCH--DCC (Capitolul 9)'),
     [T('aggregate SRISK rose before the failure of Lehman; it helps to forecast falls in industrial production and rises in unemployment', 'SRISK agregat a crescut înainte de falimentul Lehman; ajută la prognoza scăderilor producției industriale și a creșterii șomajului'),
      T('the ranking by SRISK in 2007--2008 points to the institutions that later needed public support', 'ordonarea după SRISK din 2007--2008 indică instituțiile care au avut ulterior nevoie de sprijin public')]),
    (T('Europe: \\refEJR; a lower $k = 5.5\\%$ is used because of the accounting rules (IFRS, International Financial Reporting Standards)', 'Europa: \\refEJR; se folosește un $k = 5{,}5\\%$ mai mic din cauza regulilor contabile (IFRS, International Financial Reporting Standards)'),
     [T('published and updated on \\refVLab (NYU Stern) for financial institutions from many countries', 'publicat și actualizat pe \\refVLab (NYU Stern) pentru instituții financiare din multe țări')])),
    cols(ph('acharya', 'Viral Acharya', h='0.30\\textheight'), ph('engle', 'Robert Engle', h='0.30\\textheight'), wl='0.48', wr='0.48'),
    wl='0.56', wr='0.42'), 'footnotesize')

D.recap(('MES and SRISK', 'MES și SRISK'), [
    T('MES 5\\%: the average loss of a bank on the market\'s worst 5\\% of days; LRMES $\\approx 1 - \\exp(-18\\,\\mathrm{MES})$', 'MES 5\\%: pierderea medie a unei bănci în cele mai proaste 5\\% dintre zilele pieței; LRMES $\\approx 1 - \\exp(-18\\,\\mathrm{MES})$'),
    T('SRISK $= kD - (1 - k)(1 - \\mathrm{LRMES})W$: capital missing in a crisis, driven by leverage and LRMES', 'SRISK $= kD - (1 - k)(1 - \\mathrm{LRMES})W$: capitalul lipsă într-o criză, determinat de efectul de levier și de LRMES'),
    T('MES, $\\Delta$CoVaR and VaR answer different questions and rank the banks differently', 'MES, $\\Delta$CoVaR și VaR răspund la întrebări diferite și ordonează diferit băncile')])

# =============================================================================
# 6. REȚELE
# =============================================================================
D.section('Networks of Banks', 'Rețele de bănci')

D.frame(T('The Network View', 'Perspectiva rețelei'), items(
    (T('A \\textbf{network}: nodes (banks) and directed links (``a shock to $i$ affects $j$\'\')', 'O \\textbf{rețea}: noduri (bănci) și legături orientate („un șoc la $i$ îl afectează pe $j$”)'),
     [T('real links (interbank loans) are mostly confidential; we estimate links from returns', 'legăturile reale (creditele interbancare) sînt în mare parte confidențiale; estimăm legăturile din randamente')]),
    (T('Two statistical tools', 'Două instrumente statistice'),
     [T('\\textbf{Granger causality}: do past returns of $i$ help to forecast $j$? \\refBGLP', '\\textbf{Cauzalitatea Granger}: ajută randamentele trecute ale lui $i$ la prognoza lui $j$? \\refBGLP'),
      T('\\textbf{variance decompositions}: what share of the forecast error of $j$ comes from shocks to $i$? \\refDYc', '\\textbf{descompunerea varianței}: ce parte din eroarea de prognoză a lui $j$ vine din șocurile lui $i$? \\refDYc')]),
    T('A statistical link is a co-movement, not a contract: it shows where to look, not why', 'O legătură statistică este o mișcare comună, nu un contract: arată unde trebuie căutat, nu de ce')))

D.frame(T('Granger Causality', 'Cauzalitatea Granger'), items(
    (T('$x$ \\textbf{Granger-causes} $y$ if past values of $x$ improve the forecast of $y$ beyond the past of $y$ \\refGranger', '$x$ \\textbf{cauzează în sens Granger} pe $y$ dacă valorile trecute ale lui $x$ îmbunătățesc prognoza lui $y$ dincolo de trecutul lui $y$ \\refGranger'),
     [T('regression $y_t = a + b\\,y_{t-1} + c\\,x_{t-1} + e_t$; test $H_0: c = 0$ with a $t$ test at 5\\%', 'regresia $y_t = a + b\\,y_{t-1} + c\\,x_{t-1} + e_t$; testăm $H_0: c = 0$ cu un test $t$ la 5\\%')]),
    (T('\\textbf{Degree of Granger causality} (DGC) \\refBGLP: the share of the $N(N-1)$ ordered pairs with a significant link', '\\textbf{Gradul de cauzalitate Granger} (DGC, degree of Granger causality) \\refBGLP: proporția dintre cele $N(N-1)$ perechi ordonate cu o legătură semnificativă'),
     [T('the paper: monthly returns, 36-month rolling windows, lag 1, 5\\% level', 'lucrarea: randamente lunare, ferestre mobile de 36 de luni, un decalaj, nivelul 5\\%'),
      T('the paper also filters each series with a GARCH(1,1) model; with 36 monthly observations per window we report the unfiltered test', 'lucrarea filtrează în plus fiecare serie cu un model GARCH(1,1); cu 36 de observații lunare pe fereastră, raportăm testul nefiltrat')]),
    T('Under no links, about 5\\% of pairs are significant by chance: DGC must be compared with 5\\%', 'Fără legături, aproximativ 5\\% dintre perechi sînt semnificative din întîmplare: DGC trebuie comparat cu 5\\%')))

chart(T('Granger Networks through Time', 'Rețele Granger în timp'), 'sfm_ch15_dgc', 'SFM_ch15_networks', [
    T('11 US and European banks, monthly returns since 2005 (@{gc.nm} months); first window ends in January 2008', '11 bănci americane și europene, randamente lunare din 2005 (@{gc.nm} de luni); prima fereastră se încheie în ianuarie 2008'),
    T('DGC: maximum @{gc.max}\\% (window ending @{gc.dmax}), average @{gc.mean}\\%, minimum @{gc.min}\\% (@{gc.dmin})', 'DGC: maximul @{gc.max}\\% (fereastra care se încheie pe @{gc.dmax}), media @{gc.mean}\\%, minimul @{gc.min}\\% (@{gc.dmin})')],
    h='0.58\\textheight')

chart(T('Two Networks: 2007--2009 and 2017--2019', 'Două rețele: 2007--2009 și 2017--2019'), 'sfm_ch15_granger_net', 'SFM_ch15_networks', [
    T('Arrows: significant Granger-causal links at 5\\% in a 36-month window; @{gn.2009.links} links in the crisis window, @{gn.2019.links} in the calm one',
      'Săgeți: legăturile de cauzalitate Granger semnificative la 5\\%, pe o fereastră de 36 de luni; numărul legăturilor: @{gn.2009.links} în fereastra de criză, @{gn.2019.links} în cea liniștită'),
    T('Interpretation: as in \\refBGLP, the banks became more connected around the crisis; in calm years the network is close to what chance alone would give',
      'Interpretare: ca în \\refBGLP, băncile au devenit mai conectate în jurul crizei; în anii liniștiți, rețeaua este apropiată de ceea ce ar da doar întîmplarea')],
    h='0.62\\textheight')

D.frame(T('Diebold--Yilmaz Connectedness', 'Conectivitatea Diebold--Yilmaz'), items(
    (T('Fit a VAR (vector autoregression) to the returns of $N$ banks; forecast $H$ days ahead', 'Estimăm un model VAR (vector autoregresiv) pe randamentele celor $N$ bănci; prognozăm pe $H$ zile'),
     [T('$d_{ij}$: the share of the $H$-day forecast error variance of bank $i$ due to shocks to bank $j$ (generalized decomposition, \\refPS)', '$d_{ij}$: proporția din varianța erorii de prognoză pe $H$ zile a băncii $i$ datorată șocurilor băncii $j$ (descompunerea generalizată, \\refPS)'),
      T('each row sums to 100\\%', 'fiecare rînd însumează 100\\%')]),
    (T('Measures \\refDYb; \\refDYc', 'Măsuri \\refDYb; \\refDYc'),
     [T('FROM others: $\\sum_{j \\ne i} d_{ij}$; TO others: $\\sum_{j \\ne i} d_{ji}$; NET $=$ TO $-$ FROM', 'FROM (de la celelalte): $\\sum_{j \\ne i} d_{ij}$; TO (către celelalte): $\\sum_{j \\ne i} d_{ji}$; NET $=$ TO $-$ FROM'),
      T('\\textbf{total connectedness}: $\\frac{1}{N}\\sum_{i \\ne j} d_{ij}$, the average share of risk that comes from the others', '\\textbf{conectivitatea totală}: $\\frac{1}{N}\\sum_{i \\ne j} d_{ij}$, proporția medie a riscului care vine de la celelalte bănci')]),
    T('Here: daily returns of 11 banks since 2005, VAR order by BIC (Bayesian information criterion), $H = 10$ days', 'Aici: randamentele zilnice ale celor 11 bănci din 2005, ordinul VAR ales prin BIC (Bayesian information criterion, criteriul informațional Bayesian), $H = 10$ zile')))

chart(T('The Connectedness Table', 'Tabelul de conectivitate'), 'sfm_ch15_dy_table', 'SFM_ch15_networks', [
    T('@{dyt.n} common days, VAR(@{dyt.p}); total connectedness @{dyt.total}\\%; two blocks: US banks are connected mostly with US banks, European with European',
      '@{dyt.n} zile comune, VAR(@{dyt.p}); conectivitatea totală @{dyt.total}\\%; două blocuri: băncile americane sînt conectate mai ales între ele, cele europene la fel'),
    T('Largest transmitter (TO): @{dy.topto}; smallest: @{dy.lowto} (NET $@{dyt.HSBA.net}$)', 'Cel mai mare transmițător (TO): @{dy.topto}; cel mai mic: @{dy.lowto} (NET $@{dyt.HSBA.net}$)')],
    h='0.64\\textheight')

D.frame(T('Interpretation of the Connectedness Table', 'Interpretarea tabelului de conectivitate'), items(
    (T('Each bank receives about four fifths of its forecast error variance from the others (FROM between @{dyt.HSBA.from} and @{dyt.JPM.from})', 'Fiecare bancă primește aproximativ patru cincimi din varianța erorii de prognoză de la celelalte (FROM între @{dyt.HSBA.from} și @{dyt.JPM.from})'),
     [T('daily bank returns are dominated by common shocks; the own share (the diagonal) is only about one fifth', 'randamentele zilnice ale băncilor sînt dominate de șocuri comune; partea proprie (diagonala) este doar de aproximativ o cincime')]),
    (T('NET: US banks are net transmitters (JPMorgan Chase $@{dyt.JPM.net}$), European banks net receivers (HSBC $@{dyt.HSBA.net}$)', 'NET: băncile americane sînt transmițători neți (JPMorgan Chase $@{dyt.JPM.net}$), cele europene receptori neți (HSBC $@{dyt.HSBA.net}$)'),
     [T('partly timing: New York closes after Europe, so US news reaches European prices the next day', 'parțial din cauza orelor: New York închide după Europa, deci știrile americane ajung în prețurile europene a doua zi')]),
    T('Connectedness describes the data; it is not a map of exposures between the banks', 'Conectivitatea descrie datele; nu este o hartă a expunerilor dintre bănci')))

chart(T('Total Connectedness through Time', 'Conectivitatea totală în timp'), 'sfm_ch15_dy_rolling', 'SFM_ch15_networks', [
    T('Rolling 200-day windows moved by 5 days; minimum @{dr.min}\\% (@{dr.dmin}), maximum @{dr.max}\\% (@{dr.dmax}); October 2008 -- March 2009: @{dr.y2008}\\%',
      'Ferestre mobile de 200 de zile, deplasate cu 5 zile; minimul @{dr.min}\\% (@{dr.dmin}), maximul @{dr.max}\\% (@{dr.dmax}); octombrie 2008 -- martie 2009: @{dr.y2008}\\%'),
    T('Interpretation: connectedness rose during 2007, before Lehman, and peaks in crises; it falls when bank-specific news dominate (2006, 2024)',
      'Interpretare: conectivitatea a crescut în 2007, înainte de Lehman, și atinge maximele în crize; scade cînd domină știrile specifice fiecărei bănci (2006, 2024)')],
    h='0.58\\textheight')

D.frame(T('Further Reading: the Financial Risk Meter', 'Lectură suplimentară: Financial Risk Meter'), items(
    (T('\\textbf{FRM} (Financial Risk Meter) \\refFRM: a daily index of systemic risk from quantile regressions with many regressors', '\\textbf{FRM} (Financial Risk Meter, indicatorul riscului financiar) \\refFRM: un indice zilnic al riscului sistemic, din regresii cuantilice cu mulți regresori'),
     [T('for each bank: a lasso quantile regression (Chapter 13) of its returns on the returns of all other banks and on macro variables', 'pentru fiecare bancă: o regresie cuantilică lasso (Capitolul 13) a randamentelor sale pe randamentele tuturor celorlalte bănci și pe variabile macroeconomice'),
      T('the penalty $\\lambda$ needed to fit the tail is large when the banks move together: FRM = the average $\\lambda$', 'penalizarea $\\lambda$ necesară pentru a ajusta coada este mare cînd băncile se mișcă împreună: FRM = media valorilor $\\lambda$')]),
    (T('Extensions', 'Extinderi'),
     [T('TENET (tail-event driven network) \\refTENET: CoVaR with many banks, a network of tail links', 'TENET (tail-event driven network, rețea condusă de evenimentele din coadă) \\refTENET: CoVaR cu multe bănci, o rețea de legături în coadă'),
      T('FRM for emerging markets \\refFRMem; the video course on Quantinar (Reading and Tools)', 'FRM pentru piețele emergente \\refFRMem; cursul video de pe Quantinar (Bibliografie și instrumente)')])))

D.recap(('Networks', 'rețele'), [
    T('Granger networks: share of significant links (DGC) against the 5\\% expected by chance; it rose around 2008', 'Rețelele Granger: proporția legăturilor semnificative (DGC) comparată cu 5\\%, cît dă întîmplarea; a crescut în jurul anului 2008'),
    T('Diebold--Yilmaz: shares of forecast error variance; TO, FROM, NET and the total index', 'Diebold--Yilmaz: proporții din varianța erorii de prognoză; TO, FROM, NET și indicele total'),
    T('Statistical links show co-movement in the data; they do not reveal the contracts behind it', 'Legăturile statistice arată mișcarea comună din date; nu dezvăluie contractele din spatele ei')])

# =============================================================================
# 7. POLITICA MACROPRUDENȚIALĂ
# =============================================================================
D.section('Macroprudential Policy', 'Politica macroprudențială')

D.frame(T('Micro- and Macroprudential', 'Microprudențial și macroprudențial'), items(
    T('\\textbf{Microprudential}: is each bank safe on its own? (capital, VaR, backtesting, Chapter 10)', '\\textbf{Microprudențial}: este fiecare bancă sigură luată separat? (capital, VaR, backtesting, Capitolul 10)'),
    (T('\\textbf{Macroprudential}: is the system safe? Two dimensions', '\\textbf{Macroprudențial}: este sistemul sigur? Două dimensiuni'),
     [T('\\textbf{in time}: risk builds up in booms (credit growth, leverage) and shows in busts: build buffers in good times', '\\textbf{în timp}: riscul se acumulează în perioadele de avînt (creditare, efect de levier) și devine vizibil în crize: se constituie amortizoare în perioadele bune'),
      T('\\textbf{across institutions}: larger and more connected banks need more capital', '\\textbf{între instituții}: băncile mai mari și mai interconectate au nevoie de mai mult capital')]),
    T('The measures of this chapter (CoVaR, SRISK, connectedness) are among the tools that central banks use to monitor the second dimension', 'Măsurile din acest capitol (CoVaR, SRISK, conectivitatea) se numără printre instrumentele cu care băncile centrale urmăresc a doua dimensiune')))

D.frame(T('Basel III Capital Buffers', 'Amortizoarele de capital din Basel III'), cols(items(
    (T('Minimum CET1 (common equity tier 1) capital: 4.5\\% of risk-weighted assets \\refBIII; on top of it, buffers \\refRBC:', 'Capitalul CET1 (common equity tier 1, capital comun de nivel 1) minim: 4,5\\% din activele ponderate la risc \\refBIII; peste el, amortizoare \\refRBC:'),
     [T('\\textbf{capital conservation buffer}: 2.5\\%; below it, dividends and bonuses are restricted', '\\textbf{amortizorul de conservare a capitalului}: 2,5\\%; sub el, dividendele și bonusurile sînt restricționate'),
      T('\\textbf{countercyclical buffer}: 0--2.5\\%, raised when credit grows too fast, released in a crisis', '\\textbf{amortizorul anticiclic}: 0--2,5\\%, crescut cînd creditul crește prea repede și eliberat într-o criză')]),
    (T('For systemic importance', 'Pentru importanța sistemică'),
     [T('\\textbf{G-SIB surcharge}: 1\\% to 3.5\\%, by a score of size, interconnectedness, substitutability, complexity and cross-border activity \\refGSIB',
        '\\textbf{suplimentul G-SIB}: între 1\\% și 3,5\\%, după un scor al mărimii, interconectării, substituibilității, complexității și activității transfrontaliere \\refGSIB'),
      T('in the EU: O-SII (other systemically important institutions) buffers and a systemic risk buffer', 'în UE: amortizoare O-SII (other systemically important institutions, alte instituții de importanță sistemică) și un amortizor pentru riscul sistemic')])),
    ph('bis', T('The BIS tower in Basel', 'Turnul BIS din Basel'), h='0.36\\textheight'),
    wl='0.62', wr='0.34'), 'footnotesize')

D.frame(T('Who Does It: the ESRB and the BNR', 'Cine aplică politica: ESRB și BNR'), cols(items(
    (T('\\textbf{ESRB} (European Systemic Risk Board), set up in 2010 after the crisis, chaired by the President of the ECB \\refESRB', '\\textbf{ESRB} (European Systemic Risk Board, Comitetul European pentru Risc Sistemic), înființat în 2010, după criză, condus de președintele ECB \\refESRB'),
     [T('monitors systemic risk in the EU, issues warnings and recommendations to national authorities', 'urmărește riscul sistemic în UE și emite avertismente și recomandări către autoritățile naționale')]),
    (T('Romania: \\textbf{CNSM} (Comitetul Național pentru Supravegherea Macroprudențială), with the BNR, the ASF (financial supervisory authority) and the Government \\refCNSM',
       'România: \\textbf{CNSM} (Comitetul Național pentru Supravegherea Macroprudențială), cu BNR, ASF (Autoritatea de Supraveghere Financiară) și Guvernul \\refCNSM'),
     [T('the BNR sets the capital buffers of the banks (countercyclical, O-SII, systemic risk buffer) on CNSM recommendations', 'BNR stabilește amortizoarele de capital ale băncilor (anticiclic, O-SII, pentru riscul sistemic) pe baza recomandărilor CNSM'),
      T('the BNR also uses limits on borrowers (loan-to-value, debt service-to-income) and publishes a financial stability report', 'BNR folosește și limite pentru debitori (gradul de îndatorare, raportul dintre credit și valoarea garanției) și publică un raport asupra stabilității financiare')])),
    ph('bnr', T('The BNR palace in Bucharest', 'Palatul BNR din București'), h='0.36\\textheight'),
    wl='0.60', wr='0.36'), 'footnotesize')

D.recap(('Macroprudential Policy', 'politica macroprudențială'), [
    T('Macroprudential policy targets the system: buffers built in good times, extra capital for systemic banks', 'Politica macroprudențială vizează sistemul: amortizoare constituite în perioadele bune, capital suplimentar pentru băncile sistemice'),
    T('Basel III: 4.5\\% CET1 + 2.5\\% conservation + 0--2.5\\% countercyclical + G-SIB / O-SII surcharges', 'Basel III: 4,5\\% CET1 + 2,5\\% conservare + 0--2,5\\% anticiclic + suplimente G-SIB / O-SII'),
    T('EU: the ESRB; Romania: the CNSM, with the BNR applying the capital buffers', 'UE: ESRB; România: CNSM, cu BNR aplicînd amortizoarele de capital')])

# =============================================================================
# 8. AI
# =============================================================================
D.section('AI for Scientific Discovery', 'AI pentru descoperire științifică')

D.frame(T('An Open Question', 'O întrebare deschisă'), items(
    (T('\\textbf{Is the systemic risk of the Romanian banks imported from their euro-area parents, or is it local?}', '\\textbf{Este riscul sistemic al băncilor românești importat de la băncile-mamă din zona euro sau este local?}'),
     [T('today: $\\Delta$CoVaR of TLV peaks in March 2020 ($@{dy.TLV.max}\\%$); BRD belongs to a French group, TLV is Romanian-owned', 'azi: $\\Delta$CoVaR al TLV atinge maximul în martie 2020 ($@{dy.TLV.max}\\%$); BRD aparține unui grup francez, TLV are acționariat românesc'),
      T('the BET is partly the banks themselves, so a better ``system\'\' is needed', 'BET este parțial chiar băncile, deci este nevoie de un „sistem” mai bun')]),
    T('Why it is open: two banks, few crises, different trading hours (Bucharest, Frankfurt, New York), thin trading in some years', 'Întrebarea rămîne deschisă: două bănci, puține crize, ore de tranzacționare diferite (București, Frankfurt, New York), lichiditate redusă în unii ani'),
    T('AI tools can speed up such a study; they do not replace checking it \\refWang', 'Instrumentele AI pot accelera un astfel de studiu; nu înlocuiesc verificarea lui \\refWang')))

D.frame(T('How AI Could Help', 'Contribuția posibilă a AI'), items(
    T('\\textbf{Literature}: a list of studies on systemic risk in Central and Eastern Europe and on parent--subsidiary contagion', '\\textbf{Literatura}: o listă a studiilor despre riscul sistemic în Europa Centrală și de Est și despre contagiunea dintre băncile-mamă și filiale'),
    T('\\textbf{Code}: a first draft of $\\Delta$CoVaR of TLV and BRD with the euro-area bank portfolio as the conditioning variable and state variables', '\\textbf{Cod}: o primă versiune a $\\Delta$CoVaR pentru TLV și BRD, cu portofoliul băncilor din zona euro ca variabilă de condiționare și cu variabile de stare'),
    T('\\textbf{Robustness}: two-day returns (trading hours), weekly data, CoVaR at 5\\%, other state variables', '\\textbf{Robustețe}: randamente pe două zile (orele de tranzacționare), date săptămînale, CoVaR la 5\\%, alte variabile de stare'),
    (T('Example prompt', 'Exemplu de prompt'),
     [T('\\aiprompt{Write Python code that estimates the Delta-CoVaR at 1\\% of the BET conditional on Banca Transilvania by quantile regression, with the lagged VIX and the lagged Euro Stoxx 50 return as state variables, and adds a moving-block bootstrap interval.}',
        '\\aiprompt{Write Python code that estimates the Delta-CoVaR at 1\\% of the BET conditional on Banca Transilvania by quantile regression, with the lagged VIX and the lagged Euro Stoxx 50 return as state variables, and adds a moving-block bootstrap interval.}')])))

D.frame(T('What to Check', 'Verificări necesare'), items(
    T('The level convention: ``CoVaR 1\\%\'\' and ``MES 5\\%\'\' name the tail probability; CoVaR and MES are positive losses', 'Convenția nivelului: „CoVaR 1\\%” și „MES 5\\%” indică probabilitatea cozii; CoVaR și MES sînt pierderi pozitive'),
    T('The direction of conditioning: CoVaR is the system given the bank, MES is the bank given the market', 'Direcția condiționării: CoVaR este sistemul condiționat de bancă, MES este banca condiționată de piață'),
    T('$\\Delta$CoVaR compares distress with the \\textbf{median} of the bank, not with the bank\'s VaR', '$\\Delta$CoVaR compară situația de criză cu \\textbf{mediana} băncii, nu cu VaR-ul băncii'),
    T('Prices joined on common days before returns; BVB days without trades removed; different trading hours', 'Prețurile se aliniază pe zilele comune înainte de calculul randamentelor; zilele BVB fără tranzacții se elimină; orele de tranzacționare diferă'),
    T('Uncertainty: quantile regressions at 1\\% rest on about 1\\% of the days; report bootstrap intervals', 'Incertitudinea: regresiile cuantilice la 1\\% se sprijină pe aproximativ 1\\% dintre zile; raportați intervale bootstrap'),
    T('References: every cited paper must exist; check the DOI', 'Referințele: fiecare lucrare citată trebuie să existe; verificați DOI-ul')))

D.frame(T('Project Idea', 'Idee de proiect'), items(
    (T('\\textbf{Question}: do the euro-area banks raise the tail risk of the Romanian banks in stress periods?', '\\textbf{Întrebarea}: cresc băncile din zona euro riscul din coadă al băncilor românești în perioadele de stres?'),
     [T('data: TLV, BRD, BET, the five European banks, Euro Stoxx 50, VIX, since 2010 (EODHD)', 'date: TLV, BRD, BET, cele cinci bănci europene, Euro Stoxx 50, VIX, din 2010 (EODHD)')]),
    (T('Steps', 'Pași'),
     [T('$\\Delta$CoVaR 1\\% of each Romanian bank conditional on the European bank portfolio, static and with state variables', '$\\Delta$CoVaR 1\\% al fiecărei bănci românești condiționat de portofoliul băncilor europene, static și cu variabile de stare'),
      T('MES 5\\% of TLV and BRD with the Euro Stoxx 50 as the market; Diebold--Yilmaz FROM shares on rolling windows', 'MES 5\\% pentru TLV și BRD, cu Euro Stoxx 50 ca piață; proporțiile FROM Diebold--Yilmaz pe ferestre mobile'),
      T('compare 2011--2012, March 2020 and March 2023, with block-bootstrap intervals', 'comparați 2011--2012, martie 2020 și martie 2023, cu intervale bootstrap pe blocuri')]),
    T('Deliverable: one table, one chart, and a paragraph on what the data can and cannot show', 'Livrabil: un tabel, un grafic și un paragraf despre ce pot și ce nu pot arăta datele'),
    T('Declare any AI use, and list the errors of the AI that you corrected', 'Declarați orice utilizare a instrumentelor AI și enumerați erorile acestora pe care le-ați corectat')))

# =============================================================================
# REZUMAT
# =============================================================================
D.section('Summary', 'Rezumat')

D.frame(T('Key Takeaways', 'Idei de reținut'), items(
    T('Systemic risk is a property of the system: contagion, common exposures, fire sales and runs, amplified by leverage', 'Riscul sistemic este o proprietate a sistemului: contagiune, expuneri comune, vînzări forțate și retrageri masive, amplificate de efectul de levier'),
    T('In crises bank correlations and tail dependence rise; correct for volatility before speaking of contagion', 'În crize, corelațiile și dependența în cozi dintre bănci cresc; corectați pentru volatilitate înainte de a vorbi despre contagiune'),
    T('CoVaR 1\\%: the system given the bank; $\\Delta$CoVaR $= \\hat b(\\hat q_{50\\%} - \\hat q_\\alpha)$ from quantile regression', 'CoVaR 1\\%: sistemul condiționat de bancă; $\\Delta$CoVaR $= \\hat b(\\hat q_{50\\%} - \\hat q_\\alpha)$ din regresia cuantilică'),
    T('MES 5\\%: the bank given the market; SRISK $= kD - (1 - k)(1 - \\mathrm{LRMES})W$, driven by leverage', 'MES 5\\%: banca condiționată de piață; SRISK $= kD - (1 - k)(1 - \\mathrm{LRMES})W$, determinat de efectul de levier'),
    T('Networks (Granger, Diebold--Yilmaz) show more connectedness around crises', 'Rețelele (Granger, Diebold--Yilmaz) arată o conectivitate mai mare în jurul crizelor'),
    T('Macroprudential policy: Basel III buffers, G-SIB and O-SII surcharges; the ESRB in the EU, the CNSM and the BNR in Romania', 'Politica macroprudențială: amortizoarele Basel III, suplimentele G-SIB și O-SII; ESRB în UE, CNSM și BNR în România')))

D.frame(T('Key Formulas', 'Formule de reținut'), '{\\renewcommand{\\arraystretch}{1.45}' + table(
    'll', T('\\textbf{Quantity}', '\\textbf{Mărimea}') + ' & ' + T('\\textbf{Formula}', '\\textbf{Formula}'),
    ['CoVaR & $\\mathrm{CoVaR}_\\alpha = -(\\hat a + \\hat b\\,\\hat q_\\alpha(X^i))$, \\ $X^{sys} = a + bX^i$ ' + T('at level', 'la nivelul') + ' $\\alpha$',
     '$\\Delta$CoVaR & $\\hat b\\,(\\hat q_{50\\%}(X^i) - \\hat q_\\alpha(X^i))$',
     'MES & $-E[X^i \\mid X^m \\le q_\\alpha(X^m)]$',
     'LRMES & $1 - \\exp(-18\\,\\mathrm{MES}_{2\\%})$',
     'SRISK & $kD - (1 - k)(1 - \\mathrm{LRMES})W$, \\ $L^* = 1 + (1 - k)(1 - \\mathrm{LRMES})/k$',
     'Forbes--Rigobon & $\\rho^* = \\rho/\\sqrt{1 + \\delta(1 - \\rho^2)}$',
     'Granger & $y_t = a + by_{t-1} + cx_{t-1} + e_t$, \\ $H_0: c = 0$',
     'Diebold--Yilmaz & ' + T('total', 'total') + ' $= \\frac1N\\sum_{i \\ne j} d_{ij}$, \\ NET$_i = $ TO$_i - $ FROM$_i$'],
    size='scriptsize') + '}')

D.frame(T('Check Yourself', 'Autoevaluare'), items(
    (T('\\textbf{Question}: bank A has a larger VaR 1\\% than bank B. Is A more systemic?', '\\textbf{Întrebare}: banca A are un VaR 1\\% mai mare decît banca B. Este A mai importantă sistemic?'),
     [T('\\textbf{Answer}: not necessarily: systemic importance depends on the link with the system ($\\Delta$CoVaR, MES), not on the bank\'s own tail', '\\textbf{Răspuns}: nu neapărat: importanța sistemică depinde de legătura cu sistemul ($\\Delta$CoVaR, MES), nu de coada proprie a băncii')]),
    (T('\\textbf{Question}: a quantile regression at 1\\% gives $\\hat b = 0.5$, $\\hat q_{1\\%}(X^i) = -6\\%$, $\\hat q_{50\\%}(X^i) = 0$. What is $\\Delta$CoVaR?', '\\textbf{Întrebare}: o regresie cuantilică la 1\\% dă $\\hat b = 0{,}5$, $\\hat q_{1\\%}(X^i) = -6\\%$, $\\hat q_{50\\%}(X^i) = 0$. Cît este $\\Delta$CoVaR?'),
     [T('\\textbf{Answer}: $0.5 \\times (0 - (-6)) = 3\\%$', '\\textbf{Răspuns}: $0{,}5 \\times (0 - (-6)) = 3\\%$')]),
    (T('\\textbf{Question}: why can SRISK be negative?', '\\textbf{Întrebare}: de ce poate fi SRISK negativ?'),
     [T('\\textbf{Answer}: a bank with low leverage keeps enough capital even after a market fall of 40\\%: it has a capital surplus', '\\textbf{Răspuns}: o bancă cu efect de levier mic rămîne cu suficient capital chiar și după o scădere a pieței de 40\\%: are un surplus de capital')]),
    T('Next: Chapter 16, review', 'Urmează: Capitolul 16, recapitulare')))

D.references(BIB)

if __name__ == '__main__':
    D.write(V)
