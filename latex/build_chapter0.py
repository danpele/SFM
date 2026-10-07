r"""
build_chapter0.py -- Capitolul 0 (Introducere: piețe, date, organizare), EN + RO dintr-o singură sursă
=======================================================================================================
Generator bilingv (Deck din latex/sfm_build.py, text ⟦EN||RO⟧). Cifrele vin din Quantlets/Ch_00/ch0_values.json
(Quantlets/Ch_00/generate_all_charts.py); graficele din charts/sfm_ch0_*.pdf; fotografiile din photos/ch0_*.jpg
(licențe verificate prin API-ul Wikimedia Commons; vezi photos/CREDITS.md).
Ieșire:
  EN/Courses/chapter0_introduction.tex
  RO/Cursuri/capitol0_introducere.tex
Rulare:  python3 Quantlets/Ch_00/generate_all_charts.py
         python3 latex/build_chapter0.py   apoi   python3 latex/sfm_build.py compile 0
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_build import MARK, ROOT, Deck, Values, items, cols, table, photo   # noqa: E402
from sfm_chapters import TITLES                                        # noqa: E402

VAL = json.load(open(os.path.join(ROOT, 'Quantlets', 'Ch_00', 'ch0_values.json')))

MONTHS = {'en': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                 'November', 'December'],
          'ro': ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie',
                 'octombrie', 'noiembrie', 'decembrie']}


def dt(s):
    """Data ISO -> ⟦9 March 2009||9 martie 2009⟧."""
    y, m, d = s.split('-')
    return f'⟦{int(d)} {MONTHS["en"][int(m) - 1]} {y}||{int(d)} {MONTHS["ro"][int(m) - 1]} {y}⟧'


class LangValues(Values):
    """Values whose entries may hold ⟦EN||RO⟧ text (dates): resolved for the language being rendered, since
    bilingual markers cannot be nested inside the bilingual text of a slide."""
    lang = 'en'

    def __getitem__(self, key):
        v = super().__getitem__(key)
        return MARK.sub(lambda m: m.group(1) if self.lang == 'en' else m.group(2), v) if isinstance(v, str) else v


class LangDeck(Deck):
    def frame(self, title, body, *a, **k):
        # '\\pause' given as a list element: a pause between two bullets, not an empty bullet
        super().frame(title, body.replace('    \\item \\pause\n', '    \\pause\n'), *a, **k)

    def head(self, lang):
        V.lang = lang
        return super().head(lang)


V = LangValues()
for k, x in VAL.items():
    if isinstance(x, str):
        V.raw(k, dt(x) if x[:2] in ('19', '20') and len(x) == 10 else x)
    elif k in ('sp_n', 'sp_beyond4', 'btc_vol_max_year', 'btc_vol_min_year'):
        V.int(k, x)
    elif k in ('bet_first', 'bet_last'):
        V.int(k, round(x))
    else:
        V.put(k, x, 2 if k in ('ex_r1', 'ex_r2', 'ex_rsum', 'ex_rtotal', 'sp_mean', 'sp_beyond4_normal',
                               'ratio_2025', 'sp_sd') else 1)
V.put('sp_mean3', VAL['sp_mean'], 3)
V.put('sp_sd_mean', VAL['sp_sd'] / VAL['sp_mean'], 0)
V.put('ex_ann_vol_btc252', 3.0 * VAL['ex_sqrt252'], 1)

# ---------------------------------------------------------------------------------------------------------------
# Citări (DOI verificate prin Crossref; pagini oficiale cu HTTP 200)
# ---------------------------------------------------------------------------------------------------------------
REFS = r"""
\newcommand{\refFHH}{\href{https://doi.org/10.1007/978-3-030-13751-9}{Franke, Härdle \& Hafner (2019)}}
\newcommand{\refBHL}{\href{https://doi.org/10.1007/978-3-642-33929-5}{Borak, Härdle \& López-Cabrera (2013)}}
\newcommand{\refTsay}{\href{https://doi.org/10.1002/9780470644560}{Tsay (2010)}}
\newcommand{\refCLM}{\href{https://doi.org/10.1515/9781400830213}{Campbell, Lo \& MacKinlay (1997)}}
\newcommand{\refCont}{\href{https://doi.org/10.1080/713665670}{Cont (2001)}}
\newcommand{\refBachelier}{\href{https://doi.org/10.24033/asens.476}{Bachelier (1900)}}
\newcommand{\refMandelbrot}{\href{https://doi.org/10.1086/294632}{Mandelbrot (1963)}}
\newcommand{\refFama}{\href{https://doi.org/10.2307/2325486}{Fama (1970)}}
\newcommand{\refMarkowitz}{\href{https://doi.org/10.1111/j.1540-6261.1952.tb01525.x}{Markowitz (1952)}}
\newcommand{\refBS}{\href{https://doi.org/10.1086/260062}{Black \& Scholes (1973)}}
\newcommand{\refEngle}{\href{https://doi.org/10.2307/1912773}{Engle (1982)}}
\newcommand{\refGarber}{\href{https://doi.org/10.1086/261615}{Garber (1989)}}
\newcommand{\refGoldgar}{\href{https://doi.org/10.7208/chicago/9780226301303.001.0001}{Goldgar (2007)}}
\newcommand{\refGJ}{\href{https://doi.org/10.1017/S002205070400292X}{Gelderblom \& Jonker (2004)}}
\newcommand{\refGelderblom}{\href{https://doi.org/10.23943/princeton/9780691142883.001.0001}{Gelderblom (2013)}}
\newcommand{\refFGR}{\href{https://doi.org/10.1016/j.jfineco.2012.12.008}{Frehen, Goetzmann \& Rouwenhorst (2013)}}
\newcommand{\refNeal}{\href{https://doi.org/10.1017/CBO9780511665127}{Neal (1990)}}
\newcommand{\refSWC}{\href{https://doi.org/10.1017/S0007680500000209}{Sylla, Wright \& Cowen (2009)}}
\newcommand{\refRoll}{\href{https://doi.org/10.2469/faj.v44.n5.19}{Roll (1988)}}
\newcommand{\refWhaley}{\href{https://doi.org/10.3905/jpm.2000.319728}{Whaley (2000)}}
\newcommand{\refNakamoto}{\href{https://bitcoin.org/bitcoin.pdf}{Nakamoto (2008)}}
\newcommand{\refPeleCrypto}{\href{https://doi.org/10.1080/1351847X.2021.1960403}{Pele, Wesselhöfft, Härdle \& Kolossiatis (2023)}}
\newcommand{\refPeleBVB}{\href{https://doi.org/10.1016/j.sbspro.2012.09.1030}{Pele \& Mazurencu-Marinescu (2012)}}
\newcommand{\refLM}{\href{https://doi.org/10.1093/qje/qjad055}{Ludwig \& Mullainathan (2024)}}
\newcommand{\refHLZ}{\href{https://doi.org/10.1093/rfs/hhv059}{Harvey, Liu \& Zhu (2016)}}
\newcommand{\refNBER}{\href{https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions}{NBER}}
\newcommand{\refBVB}{\href{https://www.bvb.ro/AboutUs/History}{BVB}}
\newcommand{\refSEC}{\href{https://www.sec.gov/newsroom/speeches-statements/gensler-statement-spot-bitcoin-011023}{SEC (2024)}}
\newcommand{\refCboe}{\href{https://www.cboe.com/tradable_products/vix/}{Cboe}}
"""

REFERENCES = [
    r"Bachelier, L. (1900). \href{https://doi.org/10.24033/asens.476}{Théorie de la spéculation}. \emph{Annales scientifiques de l'École normale supérieure}, 17, 21--86.",
    r"Black, F., Scholes, M. (1973). \href{https://doi.org/10.1086/260062}{The pricing of options and corporate liabilities}. \emph{Journal of Political Economy}, 81(3), 637--654.",
    r"Borak, S., Härdle, W.K., López-Cabrera, B. (2013). \href{https://doi.org/10.1007/978-3-642-33929-5}{\emph{Statistics of Financial Markets: Exercises and Solutions}} (2nd ed.). Springer.",
    r"Campbell, J.Y., Lo, A.W., MacKinlay, A.C. (1997). \href{https://doi.org/10.1515/9781400830213}{\emph{The Econometrics of Financial Markets}}. Princeton University Press.",
    r"Cont, R. (2001). \href{https://doi.org/10.1080/713665670}{Empirical properties of asset returns: stylized facts and statistical issues}. \emph{Quantitative Finance}, 1(2), 223--236.",
    r"Engle, R.F. (1982). \href{https://doi.org/10.2307/1912773}{Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation}. \emph{Econometrica}, 50(4), 987--1007.",
    r"Fama, E.F. (1970). \href{https://doi.org/10.2307/2325486}{Efficient capital markets: a review of theory and empirical work}. \emph{The Journal of Finance}, 25(2), 383--417.",
    r"Franke, J., Härdle, W.K., Hafner, C.M. (2019). \href{https://doi.org/10.1007/978-3-030-13751-9}{\emph{Statistics of Financial Markets: An Introduction}} (5th ed.). Springer.",
    r"Frehen, R.G.P., Goetzmann, W.N., Rouwenhorst, K.G. (2013). \href{https://doi.org/10.1016/j.jfineco.2012.12.008}{New evidence on the first financial bubble}. \emph{Journal of Financial Economics}, 108(3), 585--607.",
    r"Garber, P.M. (1989). \href{https://doi.org/10.1086/261615}{Tulipmania}. \emph{Journal of Political Economy}, 97(3), 535--560.",
    r"Gelderblom, O. (2013). \href{https://doi.org/10.23943/princeton/9780691142883.001.0001}{\emph{Cities of Commerce}}. Princeton University Press.",
    r"Gelderblom, O., Jonker, J. (2004). \href{https://doi.org/10.1017/S002205070400292X}{Completing a financial revolution: the finance of the Dutch East India trade and the rise of the Amsterdam capital market, 1595--1612}. \emph{The Journal of Economic History}, 64(3), 641--672.",
    r"Goldgar, A. (2007). \href{https://doi.org/10.7208/chicago/9780226301303.001.0001}{\emph{Tulipmania: Money, Honor, and Knowledge in the Dutch Golden Age}}. University of Chicago Press.",
    r"Harvey, C.R., Liu, Y., Zhu, H. (2016). \href{https://doi.org/10.1093/rfs/hhv059}{\ldots and the cross-section of expected returns}. \emph{The Review of Financial Studies}, 29(1), 5--68.",
    r"Ludwig, J., Mullainathan, S. (2024). \href{https://doi.org/10.1093/qje/qjad055}{Machine learning as a tool for hypothesis generation}. \emph{The Quarterly Journal of Economics}, 139(2), 751--827.",
    r"Mandelbrot, B. (1963). \href{https://doi.org/10.1086/294632}{The variation of certain speculative prices}. \emph{The Journal of Business}, 36(4), 394--419.",
    r"Markowitz, H. (1952). \href{https://doi.org/10.1111/j.1540-6261.1952.tb01525.x}{Portfolio selection}. \emph{The Journal of Finance}, 7(1), 77--91.",
    r"Nakamoto, S. (2008). \href{https://bitcoin.org/bitcoin.pdf}{Bitcoin: a peer-to-peer electronic cash system}. White paper.",
    r"Neal, L. (1990). \href{https://doi.org/10.1017/CBO9780511665127}{\emph{The Rise of Financial Capitalism: International Capital Markets in the Age of Reason}}. Cambridge University Press.",
    r"Pele, D.T., Mazurencu-Marinescu, M. (2012). \href{https://doi.org/10.1016/j.sbspro.2012.09.1030}{Modelling stock market crashes: the case of Bucharest Stock Exchange}. \emph{Procedia -- Social and Behavioral Sciences}, 58, 533--542.",
    r"Pele, D.T., Wesselhöfft, N., Härdle, W.K., Kolossiatis, M., Yatracos, Y.G. (2023). \href{https://doi.org/10.1080/1351847X.2021.1960403}{Are cryptos becoming alternative assets?} \emph{The European Journal of Finance}, 29(10), 1064--1105.",
    r"Roll, R. (1988). \href{https://doi.org/10.2469/faj.v44.n5.19}{The international crash of October 1987}. \emph{Financial Analysts Journal}, 44(5), 19--35.",
    r"Sylla, R., Wright, R.E., Cowen, D.J. (2009). \href{https://doi.org/10.1017/S0007680500000209}{Alexander Hamilton, central banker: crisis management during the U.S. financial panic of 1792}. \emph{Business History Review}, 83(1), 61--86.",
    r"Tsay, R.S. (2010). \href{https://doi.org/10.1002/9780470644560}{\emph{Analysis of Financial Time Series}} (3rd ed.). Wiley.",
    r"Whaley, R.E. (2000). \href{https://doi.org/10.3905/jpm.2000.319728}{The investor fear gauge}. \emph{The Journal of Portfolio Management}, 26(3), 12--17.",
]

# ---------------------------------------------------------------------------------------------------------------
# Fotografii (Wikimedia Commons; licența verificată prin API)
# ---------------------------------------------------------------------------------------------------------------
C = 'https://commons.wikimedia.org/wiki/File:'
PHOTOS = {
    'ase': ('ch0_ase_2014.jpg', C + 'Bucharest_-_Academie_de_Studii_Economice_01.jpg',
            '⟦Photo||Foto⟧: Joe Mabel (2014); CC BY 3.0; Wikimedia Commons'),
    'amsterdam': ('ch0_amsterdam_beurs_1612.jpg',
                  C + "Bird\\%27s-eye_view_of_the_Beurs_van_Hendrick_de_Keyser_by_Claes_Jansz._Visscher_(II)_1612_Stadsarchief_Amsterdam_010001000620.jpg",
                  '⟦Engraving||Gravură⟧: Claes Jansz.\\ Visscher (1612); ⟦public domain||domeniu public⟧; Stadsarchief Amsterdam, Wikimedia Commons'),
    'tulip': ('ch0_tulip_satire_1640.jpg', C + 'Jan_Brueghel_the_Younger,_Satire_on_Tulip_Mania,_c._1640.jpg',
              '⟦Painting||Pictură⟧: Jan Brueghel the Younger (c.\\ 1640); ⟦public domain||domeniu public⟧; Wikimedia Commons'),
    'southsea': ('ch0_south_sea_1721.jpg', C + 'William_Hogarth,_The_South_Sea_Scheme,_1721,_NGA_30435.jpg',
                 '⟦Print||Gravură⟧: William Hogarth (1721); CC0; National Gallery of Art, Wikimedia Commons'),
    'nyse1929': ('ch0_nyse_crowd_1929.jpg', C + 'Crowd_outside_nyse.jpg',
                 '⟦Photo||Foto⟧: U.S.\\ government (1929); ⟦public domain||domeniu public⟧; Wikimedia Commons'),
    'floor1963': ('ch0_nyse_floor_1963.jpg', C + 'NY_stock_exchange_traders_floor_LC-U9-10548-6.jpg',
                  "⟦Photo||Foto⟧: Thomas J.\\ O'Halloran (1963); ⟦public domain||domeniu public⟧; Library of Congress, Wikimedia Commons"),
    'ticker': ('ch0_edison_ticker.jpg', C + 'Thomas_Edison_stock_ticker_NMAH-JN2014-3625.jpg',
               '⟦Photo||Foto⟧: Jaclyn Nash, Smithsonian (2014); ⟦public domain||domeniu public⟧; Wikimedia Commons'),
    'bvb1928': ('ch0_bvb_palace_1928.jpg', C + 'Nicolae_Ionescu_-_The_Stock_Exchange_Palace_in_March_1928.jpg',
                '⟦Photo||Foto⟧: Nicolae Ionescu (1928); ⟦public domain||domeniu public⟧; Wikimedia Commons'),
    'bvb2023': ('ch0_bvb_palace_2023.jpg', C + 'Bucure\\%C8\\%99ti_-_Palatul_Bursei_(2023)_-_img_01.jpg',
                '⟦Photo||Foto⟧: Chainwit.\\ (2023); CC BY 4.0; Wikimedia Commons'),
    'bachelier': ('ch0_bachelier.jpg', C + 'LouisBachelier.jpg',
                  '⟦Photo||Foto⟧: ⟦unknown author||autor necunoscut⟧; ⟦public domain||domeniu public⟧; Wikimedia Commons'),
    'mandelbrot': ('ch0_mandelbrot_2007.jpg', C + 'Benoit_Mandelbrot_mg_1804-d.jpg',
                   '⟦Photo||Foto⟧: Rama (2007); CC BY-SA 2.0 fr; Wikimedia Commons'),
    'fama': ('ch0_fama_2013.jpg', C + 'Eugene_Fama_at_Nobel_Prize,_2013.jpg',
             '⟦Photo||Foto⟧: Bengt Nyman (2013); CC BY 2.0; Wikimedia Commons'),
    'lehman': ('ch0_lehman_2008.jpg', C + 'Lehman_Brothers-NYC-20080915.jpg',
               '⟦Photo||Foto⟧: Robert Scoble (2008); CC BY 2.0; Wikimedia Commons'),
    'fidi2020': ('ch0_fidi_march_2020.jpg', C + 'Subdued_FiDi_(50063555551).jpg',
                 '⟦Photo||Foto⟧: Billie Grace Ward (2020); CC BY 2.0; Wikimedia Commons'),
    'btcpaper': ('ch0_bitcoin_whitepaper.jpg', C + 'Bitcoin-whitepaper-poster_page-0001.jpg',
                 '⟦Text||Text⟧: Satoshi Nakamoto (2008); ⟦poster||afiș⟧: DailyCoinPost (2018); CC0; Wikimedia Commons'),
}


def ph(key, cap, h='0.5\\textheight'):
    f, url, cred = PHOTOS[key]
    return photo(f, cap, url, cred, h=h)


def ql(folder):
    return f'\\sfmquantlet{{Ch_00}}{{{folder}}}'


def chart(D, title, fig, folder, bullets, h='0.6\\textheight', size='footnotesize'):
    body = (f'\\begin{{center}}\n\\includegraphics[width=0.94\\textwidth,height={h},keepaspectratio]{{{fig}.pdf}}\n'
            f'\\end{{center}}\n\\vspace{{-2mm}}\n' + items(*bullets) + '\n' + ql(folder))
    D.frame(title, body, size)


QUESTION = '⟦Question for the room||Întrebare pentru sală⟧'
QUESTIONS = '⟦Questions for the room||Întrebări pentru sală⟧'
ANSWER = '⟦Answer||Răspuns⟧'

D = LangDeck(0, 'lecture', refs=REFS)

# ===============================================================================================================
D.section('Course Organisation', 'Organizarea cursului')
# ===============================================================================================================
D.frame('⟦Route of This Chapter||Traseul acestui capitol⟧', cols(
    items(('⟦\\textbf{Guiding questions}||\\textbf{Întrebări de pornire}⟧',
           ['⟦what does a statistician measure on a market?||ce măsoară un statistician pe o piață?⟧',
            '⟦which numbers describe 2026 markets?||ce cifre descriu piețele din 2026?⟧',
            '⟦how do we get from prices to returns?||cum ajungem de la prețuri la randamente?⟧']),
          ('⟦\\textbf{Part I}||\\textbf{Partea I}⟧',
           ['⟦how the course works: evaluation, textbook, chapters, AI policy||cum funcționează cursul: evaluare, manual, capitole, politica privind AI⟧',
            '⟦why financial markets matter, and what the 2026 data show||de ce contează piețele financiare și ce arată datele din 2026⟧']),
          ('⟦\\textbf{Part II}||\\textbf{Partea a II-a}⟧',
           ['⟦a short history of exchanges, crashes and models||o scurtă istorie a burselor, a crahurilor și a modelelor⟧',
            '⟦returns and first statistics, a teaser||randamente și primele statistici: o introducere⟧',
            '⟦AI for scientific discovery: one open question||AI pentru descoperire științifică: o întrebare deschisă⟧'])),
    items(('⟦\\textbf{After this chapter you can}||\\textbf{După acest capitol puteți}⟧',
           ['⟦explain how the course is organised and graded||explica felul în care este organizat și evaluat cursul⟧',
            '⟦name the main asset classes and market participants||numi principalele clase de active și participanții la piață⟧',
            '⟦read a growth chart, a risk--return chart and a drawdown chart||citi un grafic de creștere, unul risc--randament și unul de drawdown⟧',
            '⟦compute a simple and a log return from two prices||calcula un randament simplu și unul logaritmic din două prețuri⟧',
            '⟦state what must be checked in an AI-generated analysis||spune ce trebuie verificat într-o analiză generată de AI⟧']),
          '⟦Seminar 0 comes \\textbf{before} this lecture and practises the same tools on data||Seminarul 0 are loc \\textbf{înaintea} acestui curs și exersează aceleași instrumente pe date⟧')))

D.frame('⟦What Statistics of Financial Markets Studies||Obiectul statisticii piețelor financiare⟧', items(
    ('⟦\\textbf{Statistics of financial markets}: statistical methods for prices, returns and risks of traded assets||\\textbf{Statistica piețelor financiare}: metode statistice pentru prețurile, randamentele și riscurile activelor tranzacționate⟧',
     ['⟦describe: how large and how variable are returns?||descriere: cît de mari și cît de variabile sînt randamentele?⟧',
      '⟦model: which distribution, which dependence over time?||modelare: ce distribuție, ce dependență în timp?⟧',
      '⟦measure risk: how large can a loss be on a bad day?||măsurarea riscului: cît de mare poate fi pierderea într-o zi nefavorabilă?⟧']),
    ('⟦Why financial data need their own methods||De ce datele financiare cer metode proprii⟧',
     ['⟦extreme days are far more frequent than the Normal distribution predicts (\\emph{heavy tails})||zilele extreme sînt mult mai frecvente decît prevede distribuția Normală (\\emph{cozi groase})⟧',
      '⟦calm and turbulent periods alternate (\\emph{volatility clustering})||perioadele calme alternează cu cele agitate (\\emph{volatility clustering})⟧',
      '⟦markets fall together in crises: diversification weakens when it is most needed||în crize, piețele scad simultan: diversificarea își pierde efectul tocmai cînd este cea mai necesară⟧']),
    '⟦These regularities are the \\emph{stylised facts} of returns (\\refCont), studied in Chapter 2||Aceste regularități sînt \\emph{faptele stilizate} ale randamentelor (\\refCont), studiate în Capitolul 2⟧'))

D.frame('⟦The Course at a Glance||Cursul pe scurt⟧', cols(
    items(('⟦\\textbf{Programme}||\\textbf{Program}⟧',
           ["⟦Bachelor's programme Statistics and Data Science, year 3, semester 2||Licență, programul Statistică și Data Science, anul III, semestrul 2⟧",
            '⟦Faculty of Cybernetics, Statistics and Economic Informatics (CSIE), ASE Bucharest||Facultatea de Cibernetică, Statistică și Informatică Economică (CSIE), ASE București⟧',
            '⟦academic year 2026/2027||anul universitar 2026/2027⟧']),
          ('⟦\\textbf{Lecture}||\\textbf{Curs}⟧',
           ['Prof.\\ Daniel Traian Pele, \\href{mailto:danpele@ase.ro}{danpele@ase.ro}']),
          ('⟦\\textbf{Prerequisites}||\\textbf{Cunoștințe necesare}⟧',
           ['⟦probability and statistics, linear algebra, Python||probabilități și statistică, algebră liniară, Python⟧']),
          ('⟦\\textbf{Course website}||\\textbf{Site-ul cursului}⟧',
           ['\\href{https://danpele.github.io/SFM/}{danpele.github.io/SFM}: ⟦slides, seminars, notebooks, Quantlets, quizzes||slide-uri, seminarii, notebook-uri, Quantlets, quiz-uri⟧'])),
    ph('ase', '⟦The Bucharest University of Economic Studies (ASE)||Academia de Studii Economice din București (ASE)⟧', h='0.48\\textheight'),
    '0.56', '0.40'))

D.frame('⟦Evaluation||Evaluare⟧', items(
    ('⟦\\textbf{Written exam: 70\\%}||\\textbf{Examen scris: 70\\%}⟧',
     ['⟦problems on real data and short interpretation questions, as in the seminars||probleme pe date reale și întrebări scurte de interpretare, ca la seminar⟧',
      '⟦covers the lectures and the seminars of Chapters 0--16||acoperă cursurile și seminariile Capitolelor 0--16⟧']),
    ('⟦\\textbf{Team project: 20\\%}||\\textbf{Proiect de echipă: 20\\%}⟧',
     ['⟦one research question on real market data, answered with the methods of the course||o întrebare de cercetare pe date reale de piață, cu metodele cursului⟧',
      '⟦a GitHub repository, a short report, a presentation and an oral defence||un repository GitHub, un raport scurt, o prezentare și o susținere orală⟧']),
    ('⟦\\textbf{Attendance: 10\\%}||\\textbf{Prezență: 10\\%}⟧',
     ['⟦lectures and seminars||cursuri și seminarii⟧']),
    ('⟦\\textbf{Self-assessment quizzes}||\\textbf{Quiz-uri de autoevaluare}⟧',
     ['⟦one per chapter on the website: 20 questions drawn from a bank of 24, each answer explained||cîte unul pentru fiecare capitol, pe site: 20 de întrebări extrase dintr-o bancă de 24, cu explicație pentru fiecare răspuns⟧',
      '⟦practice for the exam||pregătire pentru examen⟧'])))

D.frame('⟦Textbook and Further Reading||Manual și bibliografie suplimentară⟧', items(
    ('⟦\\textbf{Main textbook}||\\textbf{Manualul de bază}⟧',
     ['\\refFHH, \\emph{Statistics of Financial Markets: An Introduction}, ⟦5th edition, Springer||ediția a 5-a, Springer⟧',
      '⟦the chapters of the course follow its parts on returns, distributions, time series, volatility and risk||capitolele cursului urmează părțile manualului dedicate randamentelor, distribuțiilor, seriilor de timp, volatilității și riscului⟧',
      '⟦its code is public as the \\href{https://github.com/QuantLet/SFE}{SFE Quantlets}||codul este public, sub forma \\href{https://github.com/QuantLet/SFE}{SFE Quantlets}⟧']),
    ('⟦\\textbf{For the seminars}||\\textbf{Pentru seminarii}⟧',
     ['\\refBHL, \\emph{Statistics of Financial Markets: Exercises and Solutions}, Springer']),
    ('⟦\\textbf{Further reading}||\\textbf{Lecturi suplimentare}⟧',
     ['\\refTsay, \\emph{Analysis of Financial Time Series}',
      '\\refCLM, \\emph{The Econometrics of Financial Markets}',
      '⟦the \\href{https://quantinar.com/course/103/statistics-of-financial-markets}{Statistics of Financial Markets} course on Quantinar, a peer-to-peer learning platform||cursul \\href{https://quantinar.com/course/103/statistics-of-financial-markets}{Statistics of Financial Markets} pe Quantinar, o platformă de învățare peer-to-peer⟧'])))


def ch_rows(rng):
    return [f'{n} & ⟦{TITLES[n][0]}||{TITLES[n][1]}⟧'.replace('α', '$\\alpha$') for n in rng]


D.frame('⟦Course Map: Chapters 0--16||Harta cursului: Capitolele 0--16⟧', cols(
    table('rl', '\\textbf{⟦No.||Nr.⟧} & \\textbf{⟦Chapter||Capitol⟧}', ch_rows(range(0, 9)), size='footnotesize'),
    table('rl', '\\textbf{⟦No.||Nr.⟧} & \\textbf{⟦Chapter||Capitol⟧}', ch_rows(range(9, 17)), size='footnotesize'),
    '0.49', '0.49') + '\n' + items(
    '⟦Data and distributions (1--6), market efficiency (7), volatility and risk (8--11), applications (12--15), review (16)||Date și distribuții (1--6), eficiența pieței (7), volatilitate și risc (8--11), aplicații (12--15), recapitulare (16)⟧'),
    size='footnotesize')

D.frame('⟦Materials and Tools||Materiale și instrumente⟧', items(
    ('⟦\\textbf{For every chapter}||\\textbf{Pentru fiecare capitol}⟧',
     ['⟦lecture slides and seminar slides, in English and in Romanian||slide-urile de curs și de seminar, în engleză și în română⟧',
      '⟦a lecture notebook and a seminar notebook (Python), opened in Google Colab with one click||un notebook de curs și unul de seminar (Python), care se deschid în Google Colab cu un singur clic⟧',
      '⟦a quiz of 20 questions||un quiz de 20 de întrebări⟧']),
    ('⟦\\textbf{Quantlets}: every chart has a public folder with code, a description file (\\texttt{Metainfo.txt}) and the chart||\\textbf{Quantlets}: fiecare grafic are un folder public cu codul, un fișier de descriere (\\texttt{Metainfo.txt}) și graficul⟧',
     ['⟦the icon under each chart opens its Quantlet on GitHub||pictograma de sub fiecare grafic deschide Quantlet-ul corespunzător pe GitHub⟧']),
    ('⟦\\textbf{Data}: daily market data from EODHD (EOD Historical Data), saved once in the course repository||\\textbf{Date}: date zilnice de piață de la EODHD (EOD Historical Data), salvate o singură dată în repository-ul cursului⟧',
     ['⟦the same numbers on every computer, until 18 September 2026||date pînă la 18 septembrie 2026, cu aceleași cifre pe orice calculator⟧',
      '⟦EUR/RON: the official reference rate of the BNR (National Bank of Romania)||EUR/RON: cursul de referință oficial al BNR⟧'])))

D.frame('⟦Seminars||Seminarii⟧', items(
    ('⟦\\textbf{Each seminar comes before its lecture}||\\textbf{Fiecare seminar are loc înaintea cursului său}⟧',
     ['⟦it starts with a short primer, so it can be followed without the lecture||începe cu o scurtă introducere, deci poate fi urmat fără curs⟧',
      '⟦the lecture then explains why the tools work and where they fail||cursul explică apoi de ce funcționează instrumentele și unde dau greș⟧']),
    ('⟦\\textbf{Two kinds of exercises}||\\textbf{Două tipuri de exerciții}⟧',
     ['⟦\\textbf{[Solved]}: full solution in the slides and in the notebook, a model to copy||\\textbf{[Rezolvat]}: rezolvarea completă în slide-uri și în notebook, un model de urmat⟧',
      '⟦\\textbf{[Proposed]}: you solve it; the solution is discussed in class||\\textbf{[Propus]}: îl rezolvați dumneavoastră; rezolvarea se discută la seminar⟧']),
    ('⟦\\textbf{Nothing is handed in}||\\textbf{Temele nu se notează}⟧',
     ['⟦homework is practice for the exam and for the project||temele servesc drept pregătire pentru examen și pentru proiect⟧',
      '⟦one exercise in each seminar asks you to find the error in an AI-generated answer||un exercițiu din fiecare seminar vă cere să găsiți greșeala dintr-un răspuns generat de AI⟧'])))

D.frame('⟦The Team Project||Proiectul de echipă⟧', items(
    '⟦\\textbf{Question}: one research question about real market data, chosen by the team||\\textbf{Întrebarea}: o întrebare de cercetare despre date reale de piață, aleasă de echipă⟧',
    ('⟦\\textbf{Deliverables}||\\textbf{Livrabile}⟧',
     ['⟦a GitHub repository whose code reproduces every number and every chart from the saved data||un repository GitHub al cărui cod reproduce fiecare cifră și fiecare grafic din datele salvate⟧',
      '⟦a short report and a presentation of the results||un raport scurt și o prezentare a rezultatelor⟧',
      '⟦a file \\texttt{AI\\_USE.md} that declares every use of AI tools||un fișier \\texttt{AI\\_USE.md}, în care este declarată fiecare utilizare a instrumentelor AI⟧']),
    ('⟦\\textbf{What we grade}||\\textbf{Criterii de evaluare}⟧',
     ['⟦a clear question, correct methods and checks, an honest interpretation||o întrebare clară, metode și verificări corecte, o interpretare onestă⟧',
      '⟦\\textbf{oral defence}: each member explains the code and the results||\\textbf{susținere orală}: fiecare membru explică codul și rezultatele⟧']),
    '⟦Project seeds appear at the end of every chapter, in the section \\emph{AI for scientific discovery}||Idei de proiect găsiți la finalul fiecărui capitol, în secțiunea \\emph{AI pentru descoperire științifică}⟧'))

D.frame('⟦AI Policy||Politica privind AI⟧', items(
    ('⟦\\textbf{AI tools are allowed and must be declared}||\\textbf{Instrumentele AI sînt permise și trebuie declarate}⟧',
     ['⟦AI: artificial intelligence; here, assistants based on an LLM (large language model) that write text and code||AI (artificial intelligence, inteligență artificială): aici, asistenți construiți pe un LLM (model lingvistic de mari dimensiuni) care scriu text și cod⟧',
      '⟦every use goes into \\texttt{AI\\_USE.md}: the tool, the request, what was kept, what was corrected||fiecare utilizare se consemnează în \\texttt{AI\\_USE.md}: instrumentul, cererea, ce s-a păstrat, ce s-a corectat⟧']),
    ('⟦\\textbf{You are responsible for every number and every reference}||\\textbf{Răspundeți pentru fiecare cifră și fiecare referință}⟧',
     ['⟦typical AI errors: invented references, wrong formulas, code that runs but computes the wrong thing||greșeli tipice ale AI: referințe inventate, formule greșite, cod care rulează, dar calculează altceva⟧',
      '⟦every reference: open its DOI (Digital Object Identifier) and check the title||fiecare referință: deschideți DOI-ul (Digital Object Identifier) și verificați titlul⟧',
      '⟦every number: recompute it with your own code from the course data||fiecare cifră: recalculați-o cu propriul cod, din datele cursului⟧']),
    ('⟦\\textbf{Oral defence}||\\textbf{Susținerea orală}⟧',
     ['⟦if you cannot explain a line of your code, it does not count as your work||o linie de cod pe care nu o puteți explica nu este considerată muncă proprie⟧'])))

D.recap(('Organisation', 'organizarea'), [
    '⟦Grade: 70\\% written exam, 20\\% team project, 10\\% attendance||Nota: 70\\% examen scris, 20\\% proiect de echipă, 10\\% prezență⟧',
    '⟦Textbook: \\refFHH; seminars: \\refBHL||Manual: \\refFHH; seminarii: \\refBHL⟧',
    '⟦Seminars come before lectures; nothing is handed in||Seminariile preced cursurile; temele nu se notează⟧',
    '⟦AI is allowed, declared in \\texttt{AI\\_USE.md}, and checked at the oral defence||AI este permis, declarat în \\texttt{AI\\_USE.md} și verificat la susținerea orală⟧'])

# ===============================================================================================================
D.section('Why Financial Markets?', 'Importanța piețelor financiare')
# ===============================================================================================================
D.frame('⟦What a Financial Market Does||Rolul unei piețe financiare⟧', items(
    '⟦\\textbf{Financial market}: a place or a system where buyers and sellers exchange claims on future cash flows (shares, bonds, currencies, contracts, tokens) at prices set by supply and demand||\\textbf{Piață financiară}: un loc sau un sistem în care cumpărătorii și vînzătorii schimbă drepturi asupra unor fluxuri de numerar viitoare (acțiuni, obligațiuni, valute, contracte, tokeni), la prețuri stabilite de cerere și ofertă⟧',
    ('⟦\\textbf{Five functions}||\\textbf{Cinci funcții}⟧',
     ['⟦\\textbf{capital allocation}: savings reach firms and governments||\\textbf{alocarea capitalului}: economiile ajung la firme și la state⟧',
      '⟦\\textbf{risk sharing}: risks move to those willing to carry them||\\textbf{împărțirea riscului}: riscurile sînt transferate celor dispuși să și le asume⟧',
      '⟦\\textbf{liquidity}: an asset can be turned into cash quickly||\\textbf{lichiditatea}: un activ poate fi transformat repede în bani⟧',
      '⟦\\textbf{price discovery}: prices gather information spread across many traders||\\textbf{descoperirea prețului}: prețurile agregă informația dispersată între mulți participanți⟧',
      '⟦\\textbf{payments and settlement}: trades are completed safely||\\textbf{plăți și decontare}: tranzacțiile sînt finalizate în siguranță⟧']),
    '⟦For a statistician, the market is a data generator: every trade leaves a price and a time||Pentru un statistician, piața este un generator de date: fiecare tranzacție lasă în urmă un preț și un moment de timp⟧'))

D.frame('⟦Who Trades||Participanții la piață⟧', cols(
    r"""\centering
\begin{tikzpicture}[font=\scriptsize, bx/.style={text=black, rounded corners, thick, align=center, minimum width=2.1cm, minimum height=0.9cm}]
\node[bx, draw=Forest, fill=Forest!8] (s) at (0,0) {⟦Savers||Economisitori⟧\\{\tiny ⟦households, pension funds||gospodării, fonduri de pensii⟧}};
\node[bx, draw=MainBlue, fill=MainBlue!8] (i) at (2.75,0) {⟦Intermediaries||Intermediari⟧\\{\tiny ⟦banks, funds, brokers||bănci, fonduri, brokeri⟧}};
\node[bx, draw=IDAred, fill=IDAred!8] (f) at (5.5,0) {⟦Issuers||Emitenți⟧\\{\tiny ⟦firms, governments||firme, state⟧}};
\node[bx, draw=Purple, fill=Purple!8, minimum width=6.6cm] (v) at (2.75,-1.7) {⟦Venues: exchanges, dealers, clearing houses||Locuri de tranzacționare: burse, dealeri, case de compensare⟧};
\draw[-{Latex}, thick, Forest] (s) -- node[above]{⟦money||bani⟧} (i);
\draw[-{Latex}, thick, Forest] (i) -- node[above]{⟦capital||capital⟧} (f);
\draw[-{Latex}, thick, IDAred] (f.north) to[bend right=30] node[above]{⟦shares, bonds||acțiuni, obligațiuni⟧} (s.north);
\draw[{Latex}-{Latex}, thick, Purple] (i) -- (v);
\end{tikzpicture}""",
    items('⟦\\textbf{Buy side}: investors who buy assets: individuals (retail), pension funds, insurers, asset managers||\\textbf{Buy side}: investitorii care cumpără active: persoane fizice (retail), fonduri de pensii, asigurători, administratori de fonduri⟧',
          ('⟦\\textbf{Passive} or \\textbf{active} investing||Investiții \\textbf{pasive} sau \\textbf{active}⟧',
           ['⟦passive: copy an index, for example through an ETF (exchange-traded fund, a fund traded like a share)||pasivă: copierea unui indice, de exemplu printr-un ETF (exchange-traded fund, fond tranzacționat ca o acțiune)⟧',
            '⟦active: try to beat the index||activă: încercarea de a obține un randament mai mare decît al indicelui⟧']),
          '⟦\\textbf{Sell side}: brokers execute orders; market makers quote a buy and a sell price all day||\\textbf{Sell side}: brokerii execută ordine; formatorii de piață (market makers) afișează toată ziua un preț de cumpărare și unul de vînzare⟧'),
    '0.50', '0.47'), size='footnotesize')

ASSET_ROWS = [
    '⟦Equities||Acțiuni⟧ & ⟦shares, equity ETFs||acțiuni, ETF-uri pe acțiuni⟧ & ⟦dividends||dividende⟧ & ⟦market, firm||piață, firmă⟧',
    '⟦Bonds||Obligațiuni⟧ & ⟦government and corporate bonds||titluri de stat, obligațiuni corporative⟧ & ⟦coupons, principal||cupoane, principal⟧ & ⟦interest rate, credit||dobîndă, credit⟧',
    '⟦Currencies (FX)||Valute (FX)⟧ & EUR/USD, EUR/RON & ⟦interest differential||diferențial de dobîndă⟧ & ⟦currency, policy||curs, politică monetară⟧',
    '⟦Commodities||Mărfuri⟧ & ⟦gold, oil||aur, petrol⟧ & ⟦none||niciunul⟧ & ⟦supply, storage||ofertă, depozitare⟧',
    '⟦Derivatives||Derivate⟧ & ⟦futures, options||futures, opțiuni⟧ & ⟦contingent payoffs||plăți condiționate⟧ & ⟦leverage, counterparty||efect de levier, contrapartidă⟧',
    '⟦Crypto assets||Active cripto⟧ & Bitcoin, Ether & ⟦none||niciunul⟧ & ⟦volatility, regulation||volatilitate, reglementare⟧',
]
D.frame('⟦Asset Classes||Clase de active⟧', table(
    '>{\\raggedright\\arraybackslash}p{2.4cm}>{\\raggedright\\arraybackslash}p{4.2cm}>{\\raggedright\\arraybackslash}p{3.0cm}>{\\raggedright\\arraybackslash}p{3.4cm}',
    '\\textbf{⟦Class||Clasă⟧} & \\textbf{⟦Examples||Exemple⟧} & \\textbf{⟦Cash flow||Flux de numerar⟧} & \\textbf{⟦Main risks||Riscuri principale⟧}',
    ASSET_ROWS, size='footnotesize') + items(
    '⟦FX (foreign exchange): the currency market; EUR/RON = lei paid for one euro||FX (foreign exchange, piața valutară): EUR/RON = lei plătiți pentru un euro⟧',
    '⟦One statistical toolbox for all of them: returns, volatility, tails and dependence||Același set de instrumente statistice pentru toate: randamente, volatilitate, cozi și dependență⟧'),
    size='footnotesize')

BOOK = r"""\centering
\begin{tikzpicture}[x=0.5cm, y=0.36cm]
\foreach \p/\q in {1/3, 2/5, 3/4} { \fill[Forest!70] (-\q, -\p) rectangle (0, -\p+0.8); \node[left, font=\tiny] at (-\q, -\p+0.4) {\q}; }
\node[right, font=\tiny] at (0.1, -0.6) {99.99}; \node[right, font=\tiny] at (0.1, -1.6) {99.98}; \node[right, font=\tiny] at (0.1, -2.6) {99.97};
\foreach \p/\q in {1/2, 2/4, 3/6} { \fill[IDAred!70] (-\q, \p) rectangle (0, \p+0.8); \node[left, font=\tiny] at (-\q, \p+0.4) {\q}; }
\node[right, font=\tiny] at (0.1, 1.4) {100.01}; \node[right, font=\tiny] at (0.1, 2.4) {100.02}; \node[right, font=\tiny] at (0.1, 3.4) {100.03};
\draw[dashed, MainBlue] (-7, 0.4) -- (2.5, 0.4);
\node[font=\scriptsize, IDAred] at (-3.5, 5) {⟦asks (sell orders)||oferte de vînzare (ask)⟧};
\node[font=\scriptsize, Forest] at (-3.5, -4.2) {⟦bids (buy orders)||oferte de cumpărare (bid)⟧};
\end{tikzpicture}"""
D.frame('⟦How a Price Is Formed: the Order Book||Formarea prețului: registrul de ordine⟧', cols(
    BOOK,
    items('⟦\\textbf{Order book}: all waiting buy orders (\\textbf{bids}) and sell orders (\\textbf{asks}), sorted by price; bars show quantities||\\textbf{Registrul de ordine}: toate ordinele de cumpărare (\\textbf{bid}) și de vînzare (\\textbf{ask}) în așteptare, ordonate după preț; barele arată cantitățile⟧',
          '⟦\\textbf{Bid--ask spread}: best ask minus best bid, here $100.01 - 99.99 = 0.02$, the cost of trading at once||\\textbf{Spread-ul bid--ask}: cel mai bun ask minus cel mai bun bid, aici $100.01 - 99.99 = 0.02$, costul unei tranzacții imediate⟧',
          '⟦A \\textbf{market order} buys or sells at once at the best prices available||Un \\textbf{ordin la piață} cumpără sau vinde imediat, la cele mai bune prețuri disponibile⟧',
          (f'\\textbf{{{QUESTIONS}}}: ⟦a market order to buy 5 units arrives||sosește un ordin la piață de cumpărare a 5 unități⟧',
           ['⟦at what average price is it filled?||la ce preț mediu se execută?⟧',
            '⟦what is the spread afterwards?||cît este spread-ul după aceea?⟧']),
          '\\pause',
          (f'\\textbf{{{ANSWER}}}',
           ['⟦2 units at 100.01 and 3 at 100.02: average $(2 \\times 100.01 + 3 \\times 100.02)/5 = 100.016$||2 unități la 100,01 și 3 la 100,02: media $(2 \\times 100.01 + 3 \\times 100.02)/5 = 100.016$⟧',
            '⟦the best ask becomes 100.02, the spread widens to 0.03: the order moved the price||cel mai bun ask devine 100,02, spread-ul crește la 0,03: ordinul a mișcat prețul⟧'])),
    '0.42', '0.55'), size='footnotesize')

D.frame('⟦Why Statistics?||Rolul statisticii⟧', items(
    ('⟦\\textbf{Tomorrow\'s price is uncertain}: we model it as a random variable||\\textbf{Prețul de mîine este incert}: îl modelăm ca pe o variabilă aleatoare⟧',
     ['⟦$P_t$: the price on day $t$; the sequence $P_1, P_2, \\ldots$ is a \\textbf{time series}||$P_t$: prețul din ziua $t$; șirul $P_1, P_2, \\ldots$ este o \\textbf{serie de timp}⟧']),
    ('⟦\\textbf{What we measure in this course}||\\textbf{Mărimi studiate în acest curs}⟧',
     ['⟦\\textbf{return}: the relative change of the price (Chapter 1)||\\textbf{randamentul}: modificarea relativă a prețului (Capitolul 1)⟧',
      '⟦\\textbf{distribution} of returns, and how heavy its tails are (Chapters 2, 3 and 5)||\\textbf{distribuția} randamentelor și cît de groase îi sînt cozile (Capitolele 2, 3 și 5)⟧',
      '⟦\\textbf{volatility}: the standard deviation of returns, and how it changes in time (Chapters 8 and 9)||\\textbf{volatilitatea}: abaterea standard a randamentelor și felul în care se schimbă în timp (Capitolele 8 și 9)⟧',
      '⟦\\textbf{risk measures}: VaR (Value at Risk) and ES (Expected Shortfall) (Chapter 10)||\\textbf{măsuri de risc}: VaR (Value at Risk, valoarea expusă la risc) și ES (Expected Shortfall) (Capitolul 10)⟧']),
    ('⟦\\textbf{The big question}: can past prices predict future returns?||\\textbf{Marea întrebare}: pot prețurile trecute prezice randamentele viitoare?⟧',
     ['⟦the efficient market hypothesis says: hardly (Chapter 7)||potrivit ipotezei pieței eficiente, aproape deloc (Capitolul 7)⟧'])))

D.recap(('Why financial markets?', 'importanța piețelor financiare'), [
    '⟦Markets allocate capital, share risk, provide liquidity and discover prices||Piețele alocă capitalul, împart riscul, asigură lichiditatea și descoperă prețurile⟧',
    '⟦Prices come from the order book: the spread is the cost of trading at once||Prețurile se formează în registrul de ordine: spread-ul este costul unei tranzacții imediate⟧',
    '⟦Six asset classes, one statistical toolbox||Șase clase de active, un singur set de instrumente statistice⟧',
    '⟦Tomorrow\'s price is a random variable: statistics describes its returns, volatility and risk||Prețul de mîine este o variabilă aleatoare: statistica îi descrie randamentele, volatilitatea și riscul⟧'])

# ===============================================================================================================
D.section('Markets in 2026: What the Data Show', 'Piețele în 2026, în cifre')
# ===============================================================================================================
DATA_ROWS = [
    'S\\&P 500 & ⟦500 large US firms||500 de firme mari din SUA⟧ & ⟦close||închidere⟧ & ⟦US trading days||zilele de tranzacționare din SUA⟧',
    'DAX & ⟦40 large German firms||40 de firme mari din Germania⟧ & ⟦close||închidere⟧ & ⟦German trading days||zilele de tranzacționare din Germania⟧',
    'BET, BET-TR & ⟦largest firms of the BVB (Bucharest Stock Exchange)||cele mai mari firme de la BVB⟧ & ⟦close||închidere⟧ & ⟦Romanian trading days||zilele de tranzacționare din România⟧',
    '⟦Gold||Aur⟧ & ⟦XAU/USD, dollars per ounce||XAU/USD, dolari pe uncie⟧ & ⟦close||închidere⟧ & ⟦weekdays||zilele lucrătoare⟧',
    'Bitcoin & BTC/USD & ⟦close at 24:00 UTC||închidere la 24:00 UTC⟧ & ⟦every day||fiecare zi⟧',
    'VIX & ⟦expected S\\&P 500 volatility||volatilitatea așteptată a S\\&P 500⟧ & ⟦close||închidere⟧ & ⟦US trading days||zilele de tranzacționare din SUA⟧',
]
D.frame('⟦Data Used in This Course||Datele folosite în curs⟧', table(
    '>{\\raggedright\\arraybackslash}p{2.0cm}>{\\raggedright\\arraybackslash}p{4.6cm}>{\\raggedright\\arraybackslash}p{2.8cm}>{\\raggedright\\arraybackslash}p{3.6cm}',
    '\\textbf{⟦Series||Serie⟧} & \\textbf{⟦What it measures||Descriere⟧} & \\textbf{⟦Price||Preț⟧} & \\textbf{⟦Calendar||Calendar⟧}',
    DATA_ROWS, size='footnotesize') + items(
    '⟦Daily data from EODHD (EOD Historical Data), saved in the course repository, until 18 September 2026||Date zilnice de la EODHD (EOD Historical Data), salvate în repository-ul cursului, pînă la 18 septembrie 2026⟧',
    '⟦BET-TR (total return): BET with dividends reinvested; UTC: Coordinated Universal Time||BET-TR (total return): BET cu dividendele reinvestite; UTC (Coordinated Universal Time): ora universală coordonată⟧',
    '⟦Each series keeps its own calendar: Bitcoin trades 365 days a year, exchanges about 252||Fiecare serie își păstrează calendarul: Bitcoin se tranzacționează 365 de zile pe an, bursele aproximativ 252⟧'),
    size='footnotesize')

chart(D, '⟦Five Markets since 2015: Growth of 100||Cinci piețe din 2015: evoluția a 100 de unități investite⟧', 'sfm_ch0_markets', 'SFM_ch0_markets', [
    '⟦Value of 100 invested in January 2015, log scale: equal vertical distances are equal percentage changes||Valoarea a 100 de unități investite în ianuarie 2015, scară logaritmică: distanțe verticale egale corespund unor variații procentuale egale⟧',
    '⟦Indices in local currency, gold and Bitcoin in USD||Indicii în moneda locală, aurul și Bitcoin în USD⟧'], h='0.62\\textheight')

MK = [('sp500', 'S\\&P 500'), ('dax', 'DAX'), ('bettr', 'BET-TR'), ('gold', '⟦Gold||Aur⟧'), ('btc', 'Bitcoin')]
D.frame('⟦Five Markets since 2015: the Numbers||Cinci piețe din 2015: cifrele⟧', cols(
    table('lrrr', '& $P_T / P_0$ & \\textbf{⟦Return||Randament⟧} & \\textbf{⟦Volatility||Volatilitate⟧}',
          [f'{lab} & $\\times$@{{{k}_mult}} & @{{{k}_ret}} & @{{{k}_vol}}' for k, lab in MK], size='footnotesize') + items('⟦$P_T / P_0$: final price over initial price; return and volatility in \\% a year||$P_T / P_0$: prețul final împărțit la prețul inițial; randamentul și volatilitatea în \\% pe an⟧'),
    items('⟦\\textbf{Return p.a.}: average daily log return times $A$, the number of trading days per year||\\textbf{Randament anual}: media randamentelor logaritmice zilnice înmulțită cu $A$, numărul de zile de tranzacționare pe an⟧',
          '⟦\\textbf{Volatility p.a.}: standard deviation of daily returns times $\\sqrt{A}$ (definitions in Part II)||\\textbf{Volatilitate anuală}: abaterea standard a randamentelor zilnice înmulțită cu $\\sqrt{A}$ (definiții în Partea a II-a)⟧',
          ('⟦\\textbf{Reading}||\\textbf{Interpretare}⟧',
           ['⟦Bitcoin: the highest return and about four times the volatility of the stock indices||Bitcoin: cel mai mare randament și o volatilitate de circa patru ori mai mare decît a indicilor bursieri⟧',
            '⟦BET-TR beat the S\\&P 500 with lower volatility over this window||în această fereastră, BET-TR a avut un randament mai mare decît S\\&P 500, cu o volatilitate mai mică⟧',
            '⟦one window of 11 years: a different start date gives a different ranking||o singură fereastră de 11 ani: o altă dată de început produce alt clasament⟧'])),
    '0.52', '0.45') + '\n' + ql('SFM_ch0_markets'), size='footnotesize')

chart(D, '⟦Risk and Return, 2015--2026||Risc și randament, 2015--2026⟧', 'sfm_ch0_risk_return', 'SFM_ch0_risk_return', [
    '⟦Each point: one market; horizontal axis volatility (log scale), vertical axis mean return, both per year||Fiecare punct: o piață; pe orizontală volatilitatea (scară logaritmică), pe verticală randamentul mediu, ambele anuale⟧'],
    h='0.66\\textheight')

D.frame('⟦Risk and Return: Interpretation||Risc și randament: interpretare⟧', items(
    ('⟦\\textbf{Stock indices cluster} around 15--22\\% volatility a year||\\textbf{Indicii bursieri se grupează} în jurul unei volatilități de 15--22\\% pe an⟧',
     ['⟦returns between @{wig20_ret}\\% (WIG20, Warsaw) and @{ndx_ret}\\% (Nasdaq 100) a year||randamente între @{wig20_ret}\\% (WIG20, Varșovia) și @{ndx_ret}\\% (Nasdaq 100) pe an⟧']),
    ('⟦\\textbf{Crypto assets are far to the right}||\\textbf{Activele cripto se află mult mai la dreapta}⟧',
     ['⟦Bitcoin @{btc_vol}\\%, Ether @{eth_vol}\\% volatility a year||volatilitate anuală: Bitcoin @{btc_vol}\\%, Ether @{eth_vol}\\%⟧']),
    ('⟦\\textbf{Long US government bonds (TLT) lost money}: @{tlt_ret}\\% a year||\\textbf{Titlurile de stat americane pe termen lung (TLT) au pierdut}: @{tlt_ret}\\% pe an⟧',
     ['⟦TLT: an ETF of US Treasury bonds with more than 20 years to maturity; interest rates rose sharply in 2022||TLT: un ETF de titluri de stat americane cu scadența peste 20 de ani; dobînzile au crescut puternic în 2022⟧']),
    (f'\\textbf{{{QUESTION}}}',
     ['⟦does the chart prove that higher risk brings higher return?||demonstrează graficul că riscul mai mare aduce randament mai mare?⟧']),
    '\\pause',
    (f'\\textbf{{{ANSWER}}}',
     ['⟦no: these are averages of one sample; a mean return over 11 years is estimated with a large error, as Chapter 1 shows||nu: sînt medii calculate pe un singur eșantion; randamentul mediu pe 11 ani este estimat cu o eroare mare, cum arată Capitolul 1⟧'])))

chart(D, '⟦Crises and Drawdowns, 2000--2026||Crize și drawdown-uri, 2000--2026⟧', 'sfm_ch0_drawdowns', 'SFM_ch0_drawdowns', [
    '⟦\\textbf{Drawdown}: the loss from the highest previous price, $DD_t = P_t / \\max_{s \\le t} P_s - 1$||\\textbf{Drawdown}: pierderea față de cel mai mare preț anterior, $DD_t = P_t / \\max_{s \\le t} P_s - 1$⟧',
    '⟦$P_t$: the price on day $t$; $\\max_{s \\le t} P_s$: the highest price up to day $t$; $DD_t \\le 0$, and $DD_t = 0$ at a new high||$P_t$: prețul din ziua $t$; $\\max_{s \\le t} P_s$: cel mai mare preț pînă în ziua $t$, inclusiv; $DD_t \\le 0$, iar $DD_t = 0$ la un nou maxim⟧'],
    h='0.56\\textheight')

D.frame('⟦Drawdowns: Interpretation||Drawdown-uri: interpretare⟧', items(
    ('⟦\\textbf{2008 global financial crisis}: the deepest fall of both indices||\\textbf{Criza financiară globală din 2008}: cea mai adîncă scădere pentru ambii indici⟧',
     ['⟦S\\&P 500: @{sp500_mdd}\\% at @{sp500_mdd_date}, from the peak of @{sp500_mdd_peak}||S\\&P 500: @{sp500_mdd}\\% pe @{sp500_mdd_date}, de la vîrful din @{sp500_mdd_peak}⟧',
      '⟦BET: @{bet_mdd}\\% at @{bet_mdd_date}, from the peak of @{bet_mdd_peak}||BET: @{bet_mdd}\\% pe @{bet_mdd_date}, de la vîrful din @{bet_mdd_peak}⟧']),
    ('⟦\\textbf{Later crises}||\\textbf{Crize ulterioare}⟧',
     ['⟦COVID-19, 2020: S\\&P 500 @{sp500_dd2020}\\%, BET @{bet_dd2020}\\%||COVID-19, 2020: S\\&P 500 @{sp500_dd2020}\\%, BET @{bet_dd2020}\\%⟧',
      '⟦rate rises, 2022: S\\&P 500 @{sp500_dd2022}\\%, BET @{bet_dd2022}\\%||creșterea dobînzilor, 2022: S\\&P 500 @{sp500_dd2022}\\%, BET @{bet_dd2022}\\%⟧']),
    (f'\\textbf{{{QUESTION}}}',
     ['⟦how long did each index need to regain its 2007 peak?||cît timp i-a trebuit fiecărui indice ca să revină la vîrful din 2007?⟧']),
    '\\pause',
    (f'\\textbf{{{ANSWER}}}',
     ['⟦S\\&P 500: until @{sp500_recovery}; BET: until @{bet_recovery}, more than 13 years||S\\&P 500: pînă la @{sp500_recovery}; BET: pînă la @{bet_recovery}, peste 13 ani⟧',
      '⟦a deep loss needs a large gain to recover: after $-80\\%$, the price must grow five times||o pierdere mare cere un cîștig și mai mare pentru revenire: după $-80\\%$, prețul trebuie să crească de cinci ori⟧'])))

chart(D, '⟦The Fear Index: VIX, 2000--2026||Indicele fricii: VIX, 2000--2026⟧', 'sfm_ch0_vix', 'SFM_ch0_vix', [
    '⟦\\textbf{VIX}: the volatility of the S\\&P 500 over the next 30 days expected by option traders, in \\% a year (\\refCboe; \\refWhaley)||\\textbf{VIX}: volatilitatea S\\&P 500 în următoarele 30 de zile, așteptată de cei care tranzacționează opțiuni, în \\% pe an (\\refCboe; \\refWhaley)⟧'],
    h='0.62\\textheight')

D.frame('⟦The VIX: Interpretation||VIX: interpretare⟧', items(
    ('⟦\\textbf{Usually calm}: mean @{vix_mean}, median @{vix_median} since 2000||\\textbf{De obicei calm}: media @{vix_mean}, mediana @{vix_median} din 2000⟧',
     ['⟦the mean is above the median: rare spikes pull it up||media este peste mediană: vîrfurile rare o ridică⟧']),
    ('⟦\\textbf{Two spikes above 80}||\\textbf{Două vîrfuri peste 80}⟧',
     ['⟦@{vix_max1_date}: @{vix_max1}, two months after the bankruptcy of Lehman Brothers||@{vix_max1_date}: @{vix_max1}, la două luni după falimentul Lehman Brothers⟧',
      '⟦@{vix_max2_date}: @{vix_max2}, the COVID-19 market panic||@{vix_max2_date}: @{vix_max2}, panica bursieră provocată de COVID-19⟧']),
    ('⟦\\textbf{Volatility comes in waves}||\\textbf{Volatilitatea evoluează în valuri}⟧',
     ['⟦high values are followed by high values, and they fade slowly||valorile mari sînt urmate de valori mari și se diminuează lent⟧',
      '⟦this \\emph{volatility clustering} is modelled with GARCH (Generalised AutoRegressive Conditional Heteroskedasticity) in Chapter 9||fenomenul de \\emph{volatility clustering} se modelează cu GARCH (Generalised AutoRegressive Conditional Heteroskedasticity) în Capitolul 9⟧']),
    '⟦On 18 September 2026 the VIX closed at @{vix_last}: a calm market||Pe 18 septembrie 2026, VIX a închis la @{vix_last}: o piață calmă⟧'))

chart(D, '⟦The Romanian Market: BET since 1997||Piața din România: BET din 1997⟧', 'sfm_ch0_bet_history', 'SFM_ch0_bet_history', [
    '⟦\\textbf{BET}: the reference index of the BVB, launched on @{bet_first_date} at @{bet_first} points (\\refBVB)||\\textbf{BET}: indicele de referință al BVB, lansat pe @{bet_first_date} la @{bet_first} de puncte (\\refBVB)⟧',
    '⟦18 September 2026: @{bet_last} points, about $\\times$@{bet_mult} in 29 years, without dividends||18 septembrie 2026: @{bet_last} de puncte, aproximativ $\\times$@{bet_mult} în 29 de ani, fără dividende⟧'],
    h='0.58\\textheight')

D.frame('⟦The Romanian Market: Interpretation||Piața din România: interpretare⟧', items(
    ('⟦\\textbf{Three phases}||\\textbf{Trei faze}⟧',
     ['⟦1997--1999: a young market loses most of its value||1997--1999: o piață tînără își pierde cea mai mare parte din valoare⟧',
      '⟦2000--2007: strong growth before the EU (European Union) accession of 2007||2000--2007: creștere puternică înainte de aderarea la UE din 2007⟧',
      '⟦2008--2009: a drawdown of @{bet_mdd_all}\\%, then a slow recovery and new highs in 2025--2026||2008--2009: un drawdown de @{bet_mdd_all}\\%, apoi o revenire lentă și noi maxime în 2025--2026⟧']),
    ('⟦\\textbf{Crashes on the BVB can be modelled}||\\textbf{Crahurile de la BVB pot fi modelate}⟧',
     ['⟦log-periodic models fitted to the BET before 2008 (\\refPeleBVB)||modele log-periodice estimate pe BET înainte de 2008 (\\refPeleBVB)⟧']),
    ('⟦\\textbf{Why BET-TR in comparisons}||\\textbf{De ce BET-TR în comparații}⟧',
     ['⟦Romanian firms pay large dividends; an index without them understates the investor\'s gain||firmele românești plătesc dividende mari; un indice fără ele subestimează cîștigul investitorului⟧'])))

D.frame('⟦Crypto Assets in 2026||Activele cripto în 2026⟧', cols(
    items('⟦\\textbf{Bitcoin}: proposed in 2008 (\\refNakamoto) as electronic cash without a bank; traded since 2010||\\textbf{Bitcoin}: propus în 2008 (\\refNakamoto) ca bani electronici fără bancă; tranzacționat din 2010⟧',
          ('⟦\\textbf{Closer to mainstream finance}||\\textbf{Mai aproape de finanțele tradiționale}⟧',
           ['⟦January 2024: the SEC (Securities and Exchange Commission, the US market regulator) approves spot Bitcoin ETFs (\\refSEC)||ianuarie 2024: SEC (Securities and Exchange Commission, autoritatea americană de supraveghere a pieței de capital) aprobă ETF-urile spot pe Bitcoin (\\refSEC)⟧',
            '⟦are cryptos becoming an alternative asset class? (\\refPeleCrypto)||devin activele cripto o clasă alternativă de active? (\\refPeleCrypto)⟧']),
          ('⟦\\textbf{Still a different statistical world}||\\textbf{Totuși, o altă lume statistică}⟧',
           ['⟦volatility @{btc_vol}\\% a year since 2015, against @{sp500_vol}\\% for the S\\&P 500||volatilitate de @{btc_vol}\\% pe an din 2015, față de @{sp500_vol}\\% pentru S\\&P 500⟧',
            '⟦trades 24 hours a day, 7 days a week: annualise with 365 days||se tranzacționează 24 de ore din 24, 7 zile din 7: anualizăm cu 365 de zile⟧']),
          '⟦Chapter 14 is about crypto assets||Capitolul 14 este dedicat activelor cripto⟧'),
    ph('btcpaper', '⟦The Bitcoin white paper, 2008||Documentul Bitcoin (white paper), 2008⟧', h='0.6\\textheight'),
    '0.62', '0.34'), size='footnotesize')

D.recap(('markets in 2026', 'piețele în 2026'), [
    '⟦Since 2015: stock indices about 15--22\\% volatility a year, Bitcoin about @{btc_vol}\\%||Din 2015: indicii bursieri au o volatilitate de circa 15--22\\% pe an, Bitcoin de circa @{btc_vol}\\%⟧',
    '⟦Drawdowns can be deep and long: BET @{bet_mdd}\\% in 2009, recovered only in 2021||Drawdown-urile pot fi adînci și lungi: BET @{bet_mdd}\\% în 2009, cu revenire abia în 2021⟧',
    '⟦Volatility comes in waves: the VIX spiked above 80 in 2008 and 2020||Volatilitatea evoluează în valuri: VIX a depășit 80 în 2008 și în 2020⟧',
    '⟦A ranking of markets depends on the window: every average is an estimate||Clasamentul piețelor depinde de fereastră: orice medie este o estimație⟧'])

# ===============================================================================================================
D.section('A Short History of Markets', 'O scurtă istorie a piețelor')
# ===============================================================================================================
TIMELINE = r"""\begin{center}
\begin{tikzpicture}[x=0.031cm, y=0.6cm, font=\tiny]
\draw[-{Latex}, thick, MainBlue] (1590,0) -- (2035,0);
\foreach \y in {1600,1700,1800,1900,2000} { \draw[MainBlue] (\y,-0.08) -- (\y,0.08); \node[below, text=black] at (\y,-0.1) {\y}; }
\foreach \y/\t/\c/\h in {1602/{⟦VOC shares, Amsterdam||acțiuni VOC, Amsterdam⟧}/Forest/0.9, 1637/{⟦tulip mania||mania lalelelor⟧}/IDAred/1.8, 1720/{⟦South Sea bubble||bula South Sea⟧}/IDAred/0.9, 1792/{⟦NYSE origins||originile NYSE⟧}/Forest/1.8, 1882/{⟦Bucharest exchange||Bursa București⟧}/Forest/0.9, 1900/{Bachelier}/MainBlue/2.7, 1929/{⟦Black Tuesday||Marțea Neagră⟧}/IDAred/1.8, 1970/{Fama}/MainBlue/2.7, 1987/{⟦Black Monday||Lunea Neagră⟧}/IDAred/1.8, 2008/{⟦Lehman; Bitcoin||Lehman; Bitcoin⟧}/IDAred/3.6, 2020/{COVID-19}/IDAred/0.9}
{ \draw[\c, thick] (\y,0) -- (\y,\h); \fill[\c] (\y,0) circle (1.2pt); \node[above, text=black, align=center] at (\y,\h) {\y\\\t}; }
\foreach \y/\t/\c/\h in {1952/{Markowitz}/MainBlue/-0.9, 1995/{⟦BVB reopens||BVB se redeschide⟧}/Forest/-1.6}
{ \draw[\c, thick] (\y,0) -- (\y,\h); \fill[\c] (\y,0) circle (1.2pt); \node[below, text=black, align=center] at (\y,\h) {\y\\\t}; }
\end{tikzpicture}
\end{center}"""
D.frame('⟦Four Centuries in One Line||Patru secole pe o singură linie⟧', TIMELINE + '\n' + items(
    '⟦\\textcolor{Forest}{Green}: exchanges; \\textcolor{IDAred}{red}: bubbles and crashes; \\textcolor{MainBlue}{blue}: statistical models of prices||\\textcolor{Forest}{Verde}: burse; \\textcolor{IDAred}{roșu}: bule și crahuri; \\textcolor{MainBlue}{albastru}: modele statistice ale prețurilor⟧',
    '⟦Exchanges, crashes and models grew together: each crisis brought new data and new questions||Bursele, crahurile și modelele au evoluat împreună: fiecare criză a adus date noi și întrebări noi⟧'),
    size='footnotesize')

D.frame('⟦The First Exchanges: from Bruges to New York||Primele burse: de la Bruges la New York⟧', cols(
    ph('amsterdam', '⟦The Amsterdam exchange of Hendrick de Keyser, 1612||Bursa din Amsterdam a lui Hendrick de Keyser, 1612⟧', h='0.52\\textheight'),
    items('⟦\\textbf{Bruges}, 13th--15th centuries: merchants meet in front of the inn of the Van der Beurse family; the name becomes \\emph{bourse}, \\emph{bursă} (\\refGelderblom)||\\textbf{Bruges}, secolele XIII--XV: negustorii se întîlnesc în fața hanului familiei Van der Beurse; numele devine \\emph{bourse}, \\emph{bursă} (\\refGelderblom)⟧',
          ('⟦\\textbf{Amsterdam}, 1602: the Dutch East India Company (VOC) issues shares that anyone can buy and resell||\\textbf{Amsterdam}, 1602: Compania Olandeză a Indiilor de Est (VOC) emite acțiuni pe care oricine le poate cumpăra și revinde⟧',
           ['⟦continuous trading, forwards and short selling follow within a decade (\\refGJ)||tranzacționarea continuă, contractele forward și vînzarea în lipsă apar în mai puțin de un deceniu (\\refGJ)⟧']),
          '⟦\\textbf{London}: from 1698, price lists published at Jonathan\'s Coffee House (\\refNeal)||\\textbf{Londra}: din 1698, liste de prețuri publicate la cafeneaua Jonathan\'s (\\refNeal)⟧',
          '⟦\\textbf{New York}, 17 May 1792: 24 brokers sign the Buttonwood Agreement, origin of the NYSE (New York Stock Exchange) (\\refSWC)||\\textbf{New York}, 17 mai 1792: 24 de brokeri semnează Buttonwood Agreement, originea NYSE (New York Stock Exchange) (\\refSWC)⟧'),
    '0.40', '0.57'), size='footnotesize')

D.frame('⟦The First Bubbles||Primele bule speculative⟧', cols(
    ph('tulip', '⟦Jan Brueghel the Younger, \\emph{Satire on Tulip Mania}, c.\\ 1640||Jan Brueghel cel Tînăr, \\emph{Satiră despre mania lalelelor}, c.\\ 1640⟧', h='0.2\\textheight')
    + '\\\\[1mm]\n' + ph('southsea', '⟦William Hogarth, \\emph{The South Sea Scheme}, 1721||William Hogarth, \\emph{Schema South Sea}, 1721⟧', h='0.2\\textheight'),
    items('⟦\\textbf{Bubble}: a price far above any reasonable value of the asset, followed by a collapse||\\textbf{Bulă speculativă}: un preț mult peste orice valoare rezonabilă a activului, urmat de o prăbușire⟧',
          ('⟦\\textbf{Tulip mania}, Holland, 1636--1637: tulip bulb contracts rise many times, then collapse in February 1637||\\textbf{Mania lalelelor}, Olanda, 1636--1637: prețurile contractelor pe bulbi de lalele cresc de multe ori, apoi se prăbușesc în februarie 1637⟧',
           ['⟦later research: a smaller episode than the legend, with few bankruptcies (\\refGarber; \\refGoldgar)||cercetările ulterioare: un episod mai mic decît legenda, cu puține falimente (\\refGarber; \\refGoldgar)⟧']),
          ('⟦\\textbf{1720}: the Mississippi Company in Paris and the South Sea Company in London rise and fall within months||\\textbf{1720}: Compania Mississippi la Paris și Compania South Sea la Londra cresc și se prăbușesc în cîteva luni⟧',
           ['⟦prices rose most for firms tied to a real innovation, Atlantic trade and insurance (\\refFGR)||prețurile au crescut cel mai mult la firmele legate de o inovație reală: comerțul atlantic și asigurările (\\refFGR)⟧']),
          '⟦Recurring ingredients: a new story, cheap credit, then forced selling||Ingrediente recurente: o poveste nouă, credit ieftin, apoi vînzări forțate⟧'),
    '0.36', '0.60'), size='footnotesize')

D.frame('⟦Black Tuesday, 1929||Marțea Neagră, 1929⟧', cols(
    ph('nyse1929', '⟦Crowd outside the NYSE, 29 October 1929||Mulțimea din fața NYSE, 29 octombrie 1929⟧', h='0.58\\textheight'),
    items('⟦\\textbf{24--29 October 1929}: the New York market crashes; 29 October is \\textbf{Black Tuesday}||\\textbf{24--29 octombrie 1929}: piața din New York se prăbușește; 29 octombrie este \\textbf{Marțea Neagră}⟧',
          ('⟦\\textbf{The Great Depression}||\\textbf{Marea Criză}⟧',
           ['⟦the US contraction that began in August 1929 lasted until March 1933 (\\refNBER)||contracția economiei americane, începută în august 1929, a durat pînă în martie 1933 (\\refNBER)⟧']),
          ('⟦\\textbf{Consequences for statistics}||\\textbf{Consecințe pentru statistică}⟧',
           ['⟦stricter rules on disclosure: firms must publish regular, audited data||reguli mai stricte de transparență: firmele trebuie să publice date regulate și auditate⟧',
            '⟦long drawdowns are not new: the BET needed 13 years after 2008||drawdown-urile lungi nu sînt o noutate: BET a avut nevoie de 13 ani după 2008⟧'])),
    '0.36', '0.60'), size='footnotesize')

D.frame('⟦From the Trading Floor to Screens||De la ringul bursei la ecrane⟧', cols(
    ph('floor1963', '⟦The NYSE trading floor, 1963||Ringul NYSE, 1963⟧', h='0.27\\textheight')
    + '\\\\[1mm]\n' + ph('ticker', '⟦A stock ticker of Thomas Edison||Un telegraf bursier (stock ticker) al lui Thomas Edison⟧', h='0.17\\textheight'),
    items(('⟦\\textbf{Stock ticker}, from 1867: prices travel by telegraph, printed on paper strips||\\textbf{Telegraful bursier (stock ticker)}, din 1867: prețurile circulă prin telegraf, tipărite pe benzi de hîrtie⟧',
           ['⟦the first time series of prices available outside the exchange||primele serii de timp de prețuri disponibile în afara bursei⟧']),
          ('⟦\\textbf{Trading floor}: brokers shout orders in the pit||\\textbf{Ringul bursei}: brokerii strigă ordinele în ring⟧',
           ['⟦prices recorded on paper by clerks||prețuri înregistrate pe hîrtie de funcționari⟧']),
          ('⟦\\textbf{Electronic trading}, from the 1970s--1990s||\\textbf{Tranzacționarea electronică}, din anii 1970--1990⟧',
           ['⟦every order and every trade is stored, to the microsecond||fiecare ordin și fiecare tranzacție sînt stocate, la nivel de microsecundă⟧',
            '⟦algorithms send a large share of the orders on major exchanges||algoritmii trimit o mare parte din ordine pe bursele importante⟧']),
          '⟦More data made new statistics possible, from daily returns to intraday volatility||Mai multe date au făcut posibile statistici noi, de la randamente zilnice la volatilitatea intrazilnică⟧'),
    '0.40', '0.57'), size='footnotesize')

D.frame('⟦The Bucharest Stock Exchange||Bursa de Valori București⟧', cols(
    ph('bvb1928', '⟦Brokers at the Stock Exchange Palace, March 1928||Brokeri la Palatul Bursei, martie 1928⟧', h='0.27\\textheight')
    + '\\\\[1mm]\n' + ph('bvb2023', '⟦The same building in 2023||Aceeași clădire în 2023⟧', h='0.16\\textheight'),
    items('⟦\\textbf{1 December 1882}: the Bucharest exchange opens under a decree of King Carol I (\\refBVB)||\\textbf{1 decembrie 1882}: bursa din București se deschide printr-un decret al regelui Carol I (\\refBVB)⟧',
          '⟦\\textbf{1948}: trading stops under the communist regime; almost 50 years without an exchange||\\textbf{1948}: tranzacționarea se oprește sub regimul comunist; aproape 50 de ani fără bursă⟧',
          '⟦\\textbf{20 November 1995}: the first session of the re-established BVB, with 6 listed companies||\\textbf{20 noiembrie 1995}: prima ședință de tranzacționare a BVB reînființate, cu 6 companii listate⟧',
          '⟦\\textbf{1997}: the BET index starts at 1\\,000 points||\\textbf{1997}: indicele BET pornește de la 1\\,000 de puncte⟧',
          '⟦\\textbf{2020}: FTSE Russell, a global index provider, upgrades Romania to Secondary Emerging market||\\textbf{2020}: FTSE Russell, un furnizor global de indici, trece România la statutul de piață emergentă secundară⟧',
          '⟦\\textbf{July 2023}: Hidroelectrica, the largest IPO (initial public offering) in BVB history||\\textbf{iulie 2023}: Hidroelectrica, cea mai mare ofertă publică inițială (IPO, initial public offering) din istoria BVB⟧'),
    '0.30', '0.66'), size='footnotesize')

D.frame('⟦Statistics Enters Finance (1/2): Random Walks and Wild Prices||Statistica intră în finanțe (1/2): mersul aleator și variațiile extreme ale prețurilor⟧', cols(
    ph('bachelier', 'Louis Bachelier (1870--1946)', h='0.32\\textheight')
    + '\\\\[1mm]\n\\raggedright\n' + items(
        '⟦\\textbf{1900}: in his doctoral thesis, Bachelier models Paris bond prices as a \\textbf{random walk} (\\refBachelier)||\\textbf{1900}: în teza de doctorat, Bachelier modelează prețurile obligațiunilor de la Paris ca un \\textbf{mers aleator} (\\refBachelier)⟧',
        '⟦price changes: independent, with the Normal distribution||variațiile prețului: independente și cu distribuție Normală⟧'),
    ph('mandelbrot', 'Benoit Mandelbrot (1924--2010)', h='0.32\\textheight')
    + '\\\\[1mm]\n\\raggedright\n' + items(
        '⟦\\textbf{1963}: Mandelbrot shows that cotton price changes have far heavier tails than the Normal distribution (\\refMandelbrot)||\\textbf{1963}: Mandelbrot arată că variațiile prețului bumbacului au cozi mult mai groase decît distribuția Normală (\\refMandelbrot)⟧',
        '⟦he proposes the $\\alpha$-stable distributions of Chapter 3||el propune distribuțiile $\\alpha$-stabile din Capitolul 3⟧'),
    '0.48', '0.48'), size='footnotesize')

D.frame('⟦Statistics Enters Finance (2/2): Risk, Efficiency, Volatility||Statistica intră în finanțe (2/2): risc, eficiență, volatilitate⟧', cols(
    items(('⟦\\textbf{1952}: Markowitz measures risk by the variance of returns and builds diversified portfolios (\\refMarkowitz)||\\textbf{1952}: Markowitz măsoară riscul prin varianța randamentelor și construiește portofolii diversificate (\\refMarkowitz)⟧',
           ['⟦risk becomes a number that can be estimated||riscul devine o mărime care poate fi estimată⟧']),
          ('⟦\\textbf{1970}: Fama defines the \\textbf{efficient market}: prices reflect all available information (\\refFama)||\\textbf{1970}: Fama definește \\textbf{piața eficientă}: prețurile reflectă toată informația disponibilă (\\refFama)⟧',
           ['⟦testable with statistics: are returns predictable? (Chapter 7)||testabilă statistic: sînt randamentele previzibile? (Capitolul 7)⟧']),
          '⟦\\textbf{1973}: Black and Scholes price options, with volatility as the key input (\\refBS)||\\textbf{1973}: Black și Scholes evaluează opțiunile, volatilitatea fiind parametrul esențial (\\refBS)⟧',
          ('⟦\\textbf{1982}: Engle models volatility that changes over time: ARCH (AutoRegressive Conditional Heteroskedasticity) (\\refEngle)||\\textbf{1982}: Engle modelează volatilitatea variabilă în timp: ARCH (AutoRegressive Conditional Heteroskedasticity) (\\refEngle)⟧',
           ['⟦the start of Chapters 8--10||punctul de plecare al Capitolelor 8--10⟧'])),
    ph('fama', '⟦Eugene Fama at the Nobel Prize ceremony, 2013||Eugene Fama la ceremonia Premiului Nobel, 2013⟧', h='0.5\\textheight'),
    '0.62', '0.34'), size='footnotesize')

D.frame('⟦Modern Crises: 1987, 2008, 2020||Crize moderne: 1987, 2008, 2020⟧', cols(
    ph('lehman', '⟦Lehman Brothers headquarters, New York, 15 September 2008||Sediul Lehman Brothers, New York, 15 septembrie 2008⟧', h='0.2\\textheight')
    + '\\\\[1mm]\n' + ph('fidi2020', '⟦An empty Financial District, New York, March 2020||Districtul Financiar din New York, gol, martie 2020⟧', h='0.15\\textheight'),
    items('⟦\\textbf{19 October 1987, Black Monday}: the Dow Jones falls 22.6\\% in one day; large falls in every major market (\\refRoll)||\\textbf{19 octombrie 1987, Lunea Neagră}: Dow Jones scade cu 22,6\\% într-o singură zi; scăderi mari pe toate piețele importante (\\refRoll)⟧',
          ('⟦\\textbf{15 September 2008}: Lehman Brothers goes bankrupt; the global financial crisis||\\textbf{15 septembrie 2008}: Lehman Brothers dă faliment; criza financiară globală⟧',
           ['⟦risk models that assumed Normal returns underestimated the losses||modelele de risc care presupuneau distribuția Normală a randamentelor au subestimat pierderile⟧']),
          ('⟦\\textbf{March 2020}: the COVID-19 crash||\\textbf{martie 2020}: crahul provocat de COVID-19⟧',
           ['⟦the S\\&P 500 log return is @{sp_min}\\% on @{sp_min_date}, its worst day since 2000||randamentul logaritmic al S\\&P 500 a fost @{sp_min}\\% pe @{sp_min_date}, cea mai slabă zi din 2000 încoace⟧']),
          '⟦Lesson for this course: extreme days are part of the data, not errors to be removed||Lecția pentru acest curs: zilele extreme fac parte din date, nu sînt erori de eliminat⟧'),
    '0.36', '0.60'), size='footnotesize')

D.recap(('a short history', 'o scurtă istorie'), [
    '⟦Exchanges since the 1600s; bubbles and crashes almost from the start||Burse din anii 1600; bule și crahuri aproape de la început⟧',
    '⟦The Bucharest exchange: 1882--1948, then again from 1995; the BET since 1997||Bursa din București: 1882--1948, apoi din nou din 1995; BET din 1997⟧',
    '⟦Models: random walk (1900), portfolio risk (1952), heavy tails (1963), efficiency (1970), changing volatility (1982)||Modele: mers aleator (1900), riscul portofoliului (1952), cozi groase (1963), eficiență (1970), volatilitate variabilă (1982)⟧',
    '⟦Each crisis showed that extreme days are more frequent than the Normal distribution predicts||Fiecare criză a arătat că zilele extreme sînt mai frecvente decît prevede distribuția Normală⟧'])

# ===============================================================================================================
D.section('From Prices to Returns: a First Look', 'De la prețuri la randamente: primii pași')
# ===============================================================================================================
D.frame('⟦Simple and Log Returns||Randamente simple și logaritmice⟧', items(
    '⟦$P_t$: the price at the end of day $t$ (for ETFs and shares, the price adjusted for dividends and splits); $P_{t-1}$: the price of the previous day||$P_t$: prețul de la sfîrșitul zilei $t$ (pentru ETF-uri și acțiuni, prețul ajustat pentru dividende și splituri); $P_{t-1}$: prețul din ziua precedentă⟧',
    ('⟦\\textbf{Simple return}: $R_t = \\dfrac{P_t}{P_{t-1}} - 1$||\\textbf{Randament simplu}: $R_t = \\dfrac{P_t}{P_{t-1}} - 1$⟧',
     ['⟦the percentage gain of one day: what the investor earns||cîștigul procentual al unei zile, adică ceea ce obține efectiv investitorul⟧']),
    ('⟦\\textbf{Log return}: $r_t = \\ln \\dfrac{P_t}{P_{t-1}} = \\ln(1 + R_t)$||\\textbf{Randament logaritmic}: $r_t = \\ln \\dfrac{P_t}{P_{t-1}} = \\ln(1 + R_t)$⟧',
     ['⟦$\\ln$: the natural logarithm; $r_t$ is the continuously compounded return of the day||$\\ln$: logaritmul natural; $r_t$ este randamentul compus continuu al zilei⟧',
      '⟦almost equal to $R_t$ for small moves; smaller for large ones||aproape egal cu $R_t$ pentru variații mici; mai mic pentru variații mari⟧']),
    ('⟦\\textbf{Worked example}: $P_0 = 100$, $P_1 = 110$, $P_2 = 99$||\\textbf{Exemplu}: $P_0 = 100$, $P_1 = 110$, $P_2 = 99$⟧',
     ['⟦$R_1 = +10\\%$, $r_1 = \\ln 1.1 = +$@{ex_r1}\\%||$R_1 = +10\\%$, $r_1 = \\ln 1.1 = +$@{ex_r1}\\%⟧',
      '⟦$R_2 = -10\\%$, $r_2 = \\ln 0.9 = $ @{ex_r2}\\%||$R_2 = -10\\%$, $r_2 = \\ln 0.9 = $ @{ex_r2}\\%⟧'])))

D.frame('⟦Which Returns Add Up?||Randamentele care se adună⟧', items(
    (f'\\textbf{{{QUESTIONS}}}: ⟦same example, $100 \\to 110 \\to 99$||același exemplu, $100 \\to 110 \\to 99$⟧',
     ['⟦what is the sum of the two simple returns?||cît este suma celor două randamente simple?⟧',
      '⟦did the investor gain, lose or break even over the two days?||a cîștigat, a pierdut sau a rămas pe loc investitorul în cele două zile?⟧']),
    '\\pause',
    (f'\\textbf{{{ANSWER}}}',
     ['⟦$R_1 + R_2 = 10\\% - 10\\% = 0\\%$, but the investor lost: $99/100 - 1 = -1\\%$||$R_1 + R_2 = 10\\% - 10\\% = 0\\%$, dar investitorul a pierdut: $99/100 - 1 = -1\\%$⟧',
      '⟦simple returns \\textbf{compound}: $(1 + R_1)(1 + R_2) - 1 = 1.1 \\times 0.9 - 1 = -1\\%$||randamentele simple \\textbf{se compun}: $(1 + R_1)(1 + R_2) - 1 = 1.1 \\times 0.9 - 1 = -1\\%$⟧',
      '⟦log returns \\textbf{add}: $r_1 + r_2 = $ @{ex_rsum}\\% $= \\ln(99/100)$||randamentele logaritmice \\textbf{se adună}: $r_1 + r_2 = $ @{ex_rsum}\\% $= \\ln(99/100)$⟧']),
    '⟦Rule: add log returns over time; compound simple returns (details in Chapter 1)||Regula: în timp, adunăm randamentele logaritmice și compunem randamentele simple (detalii în Capitolul 1)⟧'), size='footnotesize')

D.frame('⟦Mean, Volatility and Annualisation||Media, volatilitatea și anualizarea⟧', items(
    ('⟦From $n$ daily log returns $r_1, \\ldots, r_n$||Din $n$ randamente logaritmice zilnice $r_1, \\ldots, r_n$⟧',
     ['⟦\\textbf{mean}: $\\bar r = \\frac{1}{n} \\sum_{t=1}^n r_t$||\\textbf{media}: $\\bar r = \\frac{1}{n} \\sum_{t=1}^n r_t$⟧',
      '⟦\\textbf{volatility}: the standard deviation $s = \\sqrt{\\frac{1}{n-1} \\sum_{t=1}^n (r_t - \\bar r)^2}$||\\textbf{volatilitatea}: abaterea standard $s = \\sqrt{\\frac{1}{n-1} \\sum_{t=1}^n (r_t - \\bar r)^2}$⟧',
      '⟦$s$: the typical distance of a daily return from its mean; dividing by $n-1$ instead of $n$ corrects for $\\bar r$ being estimated from the same data||$s$: distanța tipică dintre un randament zilnic și media sa; împărțirea la $n-1$ în loc de $n$ corectează faptul că $\\bar r$ este estimată din aceleași date⟧']),
    ('⟦\\textbf{Annualisation} with $A$ observations per year||\\textbf{Anualizarea} cu $A$ observații pe an⟧',
     ['⟦mean: $A \\bar r$, because log returns add up; volatility: $\\sqrt{A}\\, s$, because the variances of independent days add up||media: $A \\bar r$, pentru că randamentele logaritmice se adună; volatilitatea: $\\sqrt{A}\\, s$, pentru că varianțele zilelor independente se adună⟧',
      '⟦$A \\approx 252$ for exchanges, $A = 365$ for Bitcoin||$A \\approx 252$ pentru burse, $A = 365$ pentru Bitcoin⟧']),
    ('⟦\\textbf{Worked example}||\\textbf{Exemplu}⟧',
     ['⟦a stock index with daily volatility 1\\%: $\\sqrt{252} \\times 1\\% = $ @{ex_ann_vol}\\% a year||un indice bursier cu volatilitate zilnică de 1\\%: $\\sqrt{252} \\times 1\\% = $ @{ex_ann_vol}\\% pe an⟧',
      '⟦Bitcoin with daily volatility 3\\%: $\\sqrt{365} \\times 3\\% = $ @{ex_ann_vol_btc}\\% (with 252 by mistake: @{ex_ann_vol_btc252}\\%)||Bitcoin cu volatilitate zilnică de 3\\%: $\\sqrt{365} \\times 3\\% = $ @{ex_ann_vol_btc}\\% (cu 252, greșit: @{ex_ann_vol_btc252}\\%)⟧'])), size='footnotesize')

chart(D, '⟦S\\&P 500 Daily Returns, 2000--2026||Randamentele zilnice ale S\\&P 500, 2000--2026⟧', 'sfm_ch0_returns', 'SFM_ch0_returns', [
    '⟦@{sp_n} daily log returns, from @{sp_first} to 18 September 2026||@{sp_n} randamente logaritmice zilnice, de la @{sp_first} la 18 septembrie 2026⟧'],
    h='0.62\\textheight')

D.frame('⟦Daily Returns: Interpretation||Randamentele zilnice: interpretare⟧', items(
    ('⟦\\textbf{Small on average, large in crises}||\\textbf{Mici în medie, mari în crize}⟧',
     ['⟦mean @{sp_mean3}\\% a day, standard deviation @{sp_sd}\\% a day (@{sp_ann_vol}\\% a year)||media @{sp_mean3}\\% pe zi, abaterea standard @{sp_sd}\\% pe zi (@{sp_ann_vol}\\% pe an)⟧',
      '⟦worst day: @{sp_min}\\% on @{sp_min_date}; best day: +@{sp_max}\\% on @{sp_max_date}||cea mai slabă zi: @{sp_min}\\% pe @{sp_min_date}; cea mai bună: +@{sp_max}\\% pe @{sp_max_date}⟧']),
    ('⟦\\textbf{Volatility clustering}||\\textbf{Volatility clustering}⟧',
     ['⟦large moves come in groups: 2002, 2008--2009, 2020, 2022, April 2025||variațiile mari apar grupat: 2002, 2008--2009, 2020, 2022, aprilie 2025⟧',
      '⟦the best and the worst days are close in time||cele mai bune și cele mai slabe zile sînt apropiate în timp⟧']),
    ('⟦\\textbf{The mean is hard to see}||\\textbf{Media se vede greu}⟧',
     ['⟦the daily mean is about @{sp_sd_mean} times smaller than the daily standard deviation||media zilnică este de aproximativ @{sp_sd_mean} de ori mai mică decît abaterea standard zilnică⟧',
      '⟦estimating expected returns needs long samples; estimating volatility is much easier||estimarea randamentelor așteptate cere eșantioane lungi; estimarea volatilității este mult mai ușoară⟧'])))

chart(D, '⟦Returns versus the Normal Distribution||Randamentele față de distribuția Normală⟧', 'sfm_ch0_histogram', 'SFM_ch0_returns', [
    '⟦Histogram of the S\\&P 500 daily log returns since 2000; red: the Normal density with the same mean and variance; vertical log scale||Histograma randamentelor logaritmice zilnice ale S\\&P 500 din 2000; roșu: densitatea Normală cu aceeași medie și varianță; scară verticală logaritmică⟧'],
    h='0.6\\textheight')

D.frame('⟦Heavy Tails: Interpretation||Cozi groase: interpretare⟧', items(
    ('⟦\\textbf{More extreme days than the Normal distribution allows}||\\textbf{Mai multe zile extreme decît permite distribuția Normală}⟧',
     ['⟦days with a move larger than 4 standard deviations: @{sp_beyond4} observed||zile cu o variație mai mare de 4 abateri standard: @{sp_beyond4} observate⟧',
      '⟦expected under the Normal distribution with the same mean and variance: @{sp_beyond4_normal}||număr așteptat dacă randamentele ar urma distribuția Normală cu aceeași medie și varianță: @{sp_beyond4_normal}⟧',
      '⟦the worst day, @{sp_min_date}, lies @{sp_min_z} standard deviations from the mean||cea mai slabă zi, @{sp_min_date}, se află la @{sp_min_z} abateri standard de medie⟧']),
    ('⟦\\textbf{Kurtosis} $K = \\frac{1}{n}\\sum_{t=1}^n (r_t - \\bar r)^4 / s^4$: the mean fourth power of the standardised returns||\\textbf{Coeficientul de boltire} (kurtosis) $K = \\frac{1}{n}\\sum_{t=1}^n (r_t - \\bar r)^4 / s^4$: media puterii a patra a randamentelor standardizate⟧',
     ['⟦$(r_t - \\bar r)/s$: distance from the mean in standard deviations; the fourth power lets the extreme days dominate $K$||$(r_t - \\bar r)/s$: distanța față de medie, în abateri standard; puterea a patra face ca zilele extreme să domine $K$⟧',
      '⟦Normal distribution: $K = 3$; S\\&P 500 since 2000: @{sp_kurt}; $K > 3$ means heavier tails than the Normal||distribuția Normală: $K = 3$; S\\&P 500 din 2000: @{sp_kurt}; $K > 3$ indică cozi mai groase decît la distribuția Normală⟧']),
    ('⟦\\textbf{Also more quiet days}: the histogram is taller in the centre||\\textbf{Și mai multe zile liniștite}: histograma este mai înaltă în centru⟧',
     ['⟦risk measured with the Normal distribution is too low in the tails (Chapters 2, 5 and 10)||riscul măsurat pe baza distribuției Normale este subestimat în cozi (Capitolele 2, 5 și 10)⟧'])))

D.frame('⟦Drawdown, Step by Step||Drawdown-ul, pas cu pas⟧', cols(
    table('rrrr', '$t$ & $P_t$ & ⟦peak||vîrf⟧ $M_t$ & $DD_t$',
          ['0 & 100 & 100 & $0\\%$', '1 & 120 & 120 & $0\\%$', '2 & 90 & 120 & $-25\\%$',
           '3 & 110 & 120 & $-8.3\\%$', '4 & 130 & 130 & $0\\%$', '5 & 104 & 130 & $-20\\%$'], size='footnotesize'),
    items('⟦\\textbf{Running peak}: $M_t = \\max_{s \\le t} P_s$, the highest price over the days $s = 0, \\ldots, t$||\\textbf{Vîrful curent}: $M_t = \\max_{s \\le t} P_s$, cel mai mare preț din zilele $s = 0, \\ldots, t$⟧',
          '⟦\\textbf{Drawdown}: $DD_t = P_t / M_t - 1$, never positive||\\textbf{Drawdown}: $DD_t = P_t / M_t - 1$, niciodată pozitiv⟧',
          '⟦\\textbf{Maximum drawdown} (MDD): the most negative $DD_t$; here $-25\\%$, from 120 to 90||\\textbf{Drawdown-ul maxim} (MDD, maximum drawdown): cea mai mică valoare a lui $DD_t$; aici $-25\\%$, de la 120 la 90⟧',
          ('⟦\\textbf{Why investors watch it}||\\textbf{De ce îl urmăresc investitorii}⟧',
           ['⟦it is the loss of someone who bought at the worst moment||este pierderea celui care a cumpărat în cel mai nefavorabil moment⟧',
            '⟦it ignores how long the recovery takes: report the duration too||nu arată cît durează revenirea: raportați și durata⟧'])),
    '0.38', '0.58'), size='footnotesize')

D.recap(('from prices to returns', 'de la prețuri la randamente'), [
    '⟦$R_t = P_t/P_{t-1} - 1$ compounds; $r_t = \\ln(P_t/P_{t-1})$ adds up over time||$R_t = P_t/P_{t-1} - 1$ se compune; $r_t = \\ln(P_t/P_{t-1})$ se adună în timp⟧',
    '⟦Annualise the mean with $A$ and the volatility with $\\sqrt{A}$; $A = 365$ for Bitcoin||Anualizăm media cu $A$ și volatilitatea cu $\\sqrt{A}$; $A = 365$ pentru Bitcoin⟧',
    '⟦Daily returns: volatility clustering and heavy tails (kurtosis @{sp_kurt} against 3)||Randamentele zilnice: volatility clustering și cozi groase (coeficient de boltire @{sp_kurt}, față de 3)⟧',
    '⟦Drawdown: the loss from the previous peak; the maximum drawdown summarises the worst episode||Drawdown: pierderea față de vîrful anterior; drawdown-ul maxim rezumă cel mai sever episod⟧'])

# ===============================================================================================================
D.section('AI for Scientific Discovery', 'AI pentru descoperire științifică')
# ===============================================================================================================
D.frame('⟦An Open Question||O întrebare deschisă⟧', items(
    '⟦\\textbf{Question}: has Bitcoin become less volatile since it entered mainstream finance?||\\textbf{Întrebarea}: a devenit Bitcoin mai puțin volatil de cînd a intrat în finanțele tradiționale?⟧',
    ('⟦\\textbf{Why it matters}||\\textbf{De ce contează}⟧',
     ['⟦a less volatile Bitcoin would be easier to hold in a pension fund or a bank portfolio||un Bitcoin mai puțin volatil ar fi mai ușor de deținut în portofoliul unui fond de pensii sau al unei bănci⟧',
      '⟦the spot ETFs of January 2024 opened Bitcoin to investors who buy through ordinary brokerage accounts (\\refSEC)||ETF-urile spot din ianuarie 2024 au deschis Bitcoin investitorilor care cumpără prin conturi obișnuite de brokeraj (\\refSEC)⟧']),
    ('⟦\\textbf{Why it is open}||\\textbf{De ce este deschisă}⟧',
     ['⟦only a few years of data after 2024||doar cîțiva ani de date după 2024⟧',
      '⟦other explanations exist: a calmer market overall, a larger and more liquid market||există alte explicații: o piață mai calmă în ansamblu, o piață mai mare și mai lichidă⟧',
      '⟦related evidence: cryptos and traditional assets move closer over time (\\refPeleCrypto)||rezultate conexe: activele cripto și cele tradiționale se apropie în timp (\\refPeleCrypto)⟧'])))

D.frame('⟦How AI Could Help||Contribuția posibilă a AI⟧', items(
    ('⟦\\textbf{Literature}: ask for peer-reviewed papers on Bitcoin volatility, with DOIs||\\textbf{Literatura}: cereți articole din reviste cu evaluare colegială (peer review) despre volatilitatea Bitcoin, cu DOI⟧',
     ['⟦then open every DOI yourself: AI assistants invent plausible references||apoi deschideți dumneavoastră fiecare DOI: asistenții AI inventează referințe plauzibile⟧']),
    ('⟦\\textbf{Hypotheses}: ask for three competing explanations of a fall in volatility||\\textbf{Ipoteze}: cereți trei explicații concurente pentru o scădere a volatilității⟧',
     ['⟦each explanation should predict something the others do not||fiecare explicație trebuie să prezică ceva ce celelalte nu prezic⟧']),
    ('⟦\\textbf{Code}: ask for a first draft that computes the volatility per year||\\textbf{Codul}: cereți o primă variantă care calculează volatilitatea pentru fiecare an⟧',
     ['⟦check the annualisation, the calendar and the dates line by line||verificați anualizarea, calendarul și datele, linie cu linie⟧']),
    ('⟦\\textbf{Critique}: ask the assistant to act as a hostile referee of your result||\\textbf{Critica}: cereți asistentului să joace rolul unui recenzent ostil al rezultatului dumneavoastră⟧',
     ['⟦machine learning can generate hypotheses; testing them stays a human job (\\refLM)||învățarea automată poate genera ipoteze; testarea lor rămîne sarcina cercetătorului (\\refLM)⟧'])))

chart(D, '⟦Mini-Case: Volatility per Year||Mini-studiu de caz: volatilitatea pe ani⟧', 'sfm_ch0_btc_vol', 'SFM_ch0_btc_vol', [
    '⟦Annualised volatility of daily log returns in each calendar year; Bitcoin with 365 days, the S\\&P 500 with its own frequency; 2026*: until 18 September||Volatilitatea anualizată a randamentelor logaritmice zilnice în fiecare an calendaristic; Bitcoin cu 365 de zile, S\\&P 500 cu frecvența proprie; 2026*: pînă la 18 septembrie⟧',
    '⟦Bitcoin: average @{btc_vol_pre}\\% a year in 2015--2023, @{btc_vol_post}\\% in 2024--2026; in 2025, @{ratio_2025} times the S\\&P 500||Bitcoin: în medie @{btc_vol_pre}\\% pe an în 2015--2023, @{btc_vol_post}\\% în 2024--2026; în 2025, de @{ratio_2025} ori volatilitatea S\\&P 500⟧'],
    h='0.52\\textheight')

D.frame('⟦What to Check||Verificări necesare⟧', items(
    ('⟦\\textbf{Fix the comparison before looking at the chart}||\\textbf{Stabiliți comparația înainte de a vă uita la grafic}⟧',
     ['⟦break date: the ETF approval of January 2024, not the year that looks best||data de ruptură: aprobarea ETF-urilor din ianuarie 2024, nu anul care arată cel mai bine⟧',
      '⟦trying many dates and keeping the best one produces false discoveries (\\refHLZ)||dacă încercați multe date și o păstrați pe cea mai bună, obțineți descoperiri false (\\refHLZ)⟧']),
    ('⟦\\textbf{Rule out the rival explanation}||\\textbf{Excludeți explicația concurentă}⟧',
     ['⟦if all markets were calmer, the S\\&P 500 volatility fell too: compare the ratio Bitcoin / S\\&P 500||dacă toate piețele au fost mai calme, a scăzut și volatilitatea S\\&P 500: comparați raportul Bitcoin / S\\&P 500⟧']),
    ('⟦\\textbf{Check the numbers}||\\textbf{Verificați cifrele}⟧',
     ['⟦365 days a year for Bitcoin; the same definition of volatility in every year||365 de zile pe an pentru Bitcoin; aceeași definiție a volatilității în fiecare an⟧',
      '⟦one year is a short sample: a single calm year is not a trend||un an este un eșantion scurt: un singur an calm nu este o tendință⟧']),
    '⟦\\textbf{Say what the result cannot show}: a fall after 2024 does not prove that the ETFs caused it||\\textbf{Precizați ce nu poate arăta rezultatul}: o scădere după 2024 nu dovedește că ETF-urile au cauzat-o⟧'))

D.frame('⟦Your Turn: a Project Seed||Rîndul dumneavoastră: o idee de proiect⟧', items(
    '⟦\\textbf{Question}: is Bitcoin\'s volatility, relative to the S\\&P 500, lower after January 2024 than before?||\\textbf{Întrebarea}: este volatilitatea Bitcoin, raportată la S\\&P 500, mai mică după ianuarie 2024 decît înainte?⟧',
    ('⟦\\textbf{Steps}||\\textbf{Pașii}⟧',
     ['⟦replicate the chart of the mini-case from the course data||reproduceți graficul mini-studiului de caz din datele cursului⟧',
      '⟦write down the hypothesis, the break date and the test before computing||scrieți ipoteza, data de ruptură și testul înainte de calcul⟧',
      '⟦compare the variances of daily returns before and after with a test for equal variances (Levene test)||comparați varianțele randamentelor zilnice înainte și după, cu un test de egalitate a varianțelor (testul Levene)⟧',
      '⟦repeat with Ether as a second crypto asset||repetați cu Ether, ca al doilea activ cripto⟧']),
    ('⟦\\textbf{Report}||\\textbf{Raportați}⟧',
     ['⟦the two volatilities, the ratio, the test result and one limitation||cele două volatilități, raportul, rezultatul testului și o limită⟧',
      '⟦every AI prompt that shaped a decision, in \\texttt{AI\\_USE.md}||fiecare prompt AI care a influențat o decizie, în \\texttt{AI\\_USE.md}⟧'])))

# ===============================================================================================================
D.section('Conclusions', 'Concluzii')
# ===============================================================================================================
D.frame('⟦Key Takeaways||Idei principale⟧', items(
    '⟦Grade: 70\\% exam, 20\\% project, 10\\% attendance; AI allowed and declared; seminars before lectures||Nota: 70\\% examen, 20\\% proiect, 10\\% prezență; AI permis și declarat; seminariile înaintea cursurilor⟧',
    '⟦Financial markets allocate capital and risk; their prices are the data of this course||Piețele financiare alocă capitalul și riscul; prețurile lor sînt datele acestui curs⟧',
    ('⟦\\textbf{2000--2026 in numbers}||\\textbf{2000--2026 în cifre}⟧',
     ['⟦stock indices: about 15--22\\% volatility a year; Bitcoin: about @{btc_vol}\\%||indicii bursieri: circa 15--22\\% volatilitate pe an; Bitcoin: circa @{btc_vol}\\%⟧',
      '⟦drawdowns of @{sp500_mdd}\\% (S\\&P 500) and @{bet_mdd}\\% (BET) in 2008--2009||drawdown-uri de @{sp500_mdd}\\% (S\\&P 500) și @{bet_mdd}\\% (BET) în 2008--2009⟧']),
    '⟦Four centuries of bubbles and crashes; models moved from the Normal random walk to heavy tails and changing volatility||Patru secole de bule și crahuri; modelele au trecut de la mersul aleator Normal la cozi groase și volatilitate variabilă⟧',
    '⟦Log returns add, simple returns compound; volatility scales with $\\sqrt{A}$||Randamentele logaritmice se adună, cele simple se compun; volatilitatea se anualizează cu factorul $\\sqrt{A}$⟧',
    '⟦An AI-assisted result counts only after its references, numbers and rival explanations are checked||Un rezultat obținut cu ajutorul AI contează doar după ce referințele, cifrele și explicațiile concurente sînt verificate⟧'))

D.frame('⟦Check Yourself, and Next: Chapter 1||Autoevaluare; urmează Capitolul 1⟧', cols(
    items(('⟦\\textbf{Check yourself}||\\textbf{Autoevaluare}⟧',
           ['⟦Why is the sum of simple returns not the total return?||De ce suma randamentelor simple nu este randamentul total?⟧',
            '⟦How do you annualise a daily volatility of Bitcoin?||Cum anualizați volatilitatea zilnică a Bitcoin?⟧',
            '⟦What does a maximum drawdown of $-80\\%$ mean for an investor?||Ce înseamnă pentru un investitor un drawdown maxim de $-80\\%$?⟧',
            '⟦What must you check before using an AI-generated reference?||Ce trebuie să verificați înainte de a folosi o referință generată de AI?⟧'])),
    items(('⟦\\textbf{Chapter 1: data, returns and indicators}||\\textbf{Capitolul 1: date, randamente și indicatori}⟧',
           ['⟦Where do the data come from?||De unde vin datele?⟧',
            '⟦Which price field should we use?||Ce coloană de preț folosim?⟧',
            '⟦How do returns aggregate over days, months and years?||Cum se agregă randamentele pe zile, luni și ani?⟧',
            '⟦What is the Sharpe ratio?||Ce este raportul Sharpe?⟧',
            '⟦What is the volatility drag?||Ce înseamnă volatility drag?⟧'])),
    '0.48', '0.48'))

D.references(REFERENCES, per=13)

if __name__ == '__main__':
    D.write(V)
