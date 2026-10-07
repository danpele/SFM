r"""
build_chapter14.py -- Capitolul 14 (Active cripto: criptomonede și stablecoins), EN + RO dintr-o singură sursă
=============================================================================================================
Text ⟦english||română⟧; cifrele @{cheie} vin din Quantlets/Ch_14/ch14_numbers.json (generate_all_charts.py) sau
sînt calculate aici, în Python, pentru exemplele lucrate. Nicio cifră nu este scrisă de mînă.
Date: închideri zilnice EODHD (Bitcoin, Ethereum, Solana, USDT, USDC, DAI, S&P 500, aur, ETF-ul IBIT) și oferta
de stablecoins de la DefiLlama.
Ieșire:
  EN/Courses/chapter14_crypto_assets.tex
  RO/Cursuri/capitol14_active_cripto.tex
Rulare:
  python3 Quantlets/Ch_14/generate_all_charts.py
  python3 latex/build_chapter14.py && python3 latex/sfm_build.py compile 14
"""

import math
import os
import sys

from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_build import Deck, cols, items, table, photo   # noqa: E402
from ch14_common import BIB, QLURL, REFS, T, load, values, money   # noqa: E402

N = load()
V = values(N)
D = Deck(14, 'lecture', refs=REFS)
C = 'https://commons.wikimedia.org/wiki/File:'


def ql(folder):
    return f'\\sfmquantlet{{Ch_14}}{{{folder}}}'


def chart(title, fig, folder, bullets, h='0.56\\textheight', size='footnotesize'):
    body = (f'\\begin{{center}}\n\\includegraphics[width=0.97\\textwidth,height={h},keepaspectratio]{{{fig}.pdf}}\n'
            f'\\end{{center}}\n\\vspace{{-0.25cm}}\n' + items(*bullets) + '\n' + ql(folder))
    D.frame(title, body, size)


PH = {
    'paper': ('ch0_bitcoin_whitepaper.jpg', C + 'Bitcoin-whitepaper-poster_page-0001.jpg',
              '⟦Text||Text⟧: Satoshi Nakamoto (2008); ⟦poster||afiș⟧: DailyCoinPost (2018); CC0; Wikimedia Commons'),
    'bust': ('ch14_satoshi_bust_2021.jpg', C + 'Bust_of_Satoshi_Nakamoto_in_Budapest.jpg',
             '⟦Photo||Foto⟧: Fekist (2021); CC BY-SA 4.0; Wikimedia Commons'),
    'mining': ('ch14_mining_farm_2014.jpg', C + 'Bitcoin_mining_farm.jpg',
               '⟦Photo||Foto⟧: Marko Ahtisaari (2014); CC BY 2.0; Wikimedia Commons'),
    'buterin': ('ch14_buterin_2015.jpg', C + 'Vitalik_Buterin_TechCrunch_London_2015_(cropped).jpg',
                '⟦Photo||Foto⟧: John Phillips (2015); CC BY 2.0; Wikimedia Commons'),
    'atm': ('ch11_bitcoin_atm_prague.jpg', C + 'Bitcoin_ATM_Prague.jpg', '⟦Photo||Foto⟧: Perituss (2016); CC0; Wikimedia Commons'),
    'sec': ('ch14_sec_hq_2008.jpg', C + 'Facade_of_the_U.S._Securities_and_Exchange_Commission_headquarters,_Washington,_D.C.jpg',
            '⟦Photo||Foto⟧: David (dbking) (2008); CC BY 2.0; Wikimedia Commons'),
    'svb': ('ch14_svb_hq_2023.jpg', C + '3003_West_Tasman_Drive_entrance_2,_Santa_Clara,_California.jpg',
            '⟦Photo||Foto⟧: Minh Nguyen (2023); CC BY-SA 4.0; Wikimedia Commons'),
    'ep': ('ch14_eu_parliament_2010.jpg', C + 'Hemicycle_of_Louise_Weiss_building_of_the_European_Parliament,_Strasbourg.jpg',
           '⟦Photo||Foto⟧: jeffowenphotos (2010); CC BY 2.0; Wikimedia Commons'),
    'capitol': ('ch14_us_capitol.jpg', C + 'United_States_Capitol_-_west_front.jpg',
                '⟦Photo||Foto⟧: Architect of the Capitol; ⟦public domain||domeniu public⟧; Wikimedia Commons'),
    'sornette': ('ch14_sornette_2012.jpg', C + 'Didier_Sornette.jpg', '⟦Photo||Foto⟧: Didier Sornette (2012); CC BY-SA 3.0 DE; Wikimedia Commons'),
}


def ph(key, cap, h='0.48\\textheight'):
    f, url, cred = PH[key]
    return photo(f, cap, url, cred, h=h)


# =============================================================================
# EXEMPLE LUCRATE (calculate aici)
# =============================================================================
s = N['stats']
# annualisation: Bitcoin, sqrt(365) against sqrt(252)
V.put('an.s365', math.sqrt(365), 2)
V.put('an.s252', math.sqrt(252), 2)
V.put('an.ratio', math.sqrt(365 / 252), 3)
V.put('an.under', 100 * (1 - math.sqrt(252 / 365)), 1)
V.put('an.underm', 100 * (1 - 252 / 365), 1)
# drawdown: a fall of 80% needs a gain of 400%
V.put('dd.ex80', 100 * (1 / 0.20 - 1), 0)
V.put('dd.btc.lastabs', -N['dd']['btc']['last'], 1)
ep = N['dd']['btc_episodes']
big = min(ep, key=lambda e: e['depth'])
V.put('dd.big.need', 100 * (1 / (1 + big['depth'] / 100) - 1), 0)
# peg deviation: USDC on 11 March 2023
sv = N['svb']['usdc']
V.raw('svb.loss', money(1_000_000 * (1 - sv['close_min'])))
V.raw('svb.losslow', money(1_000_000 * (1 - sv['low'])))
# VaR in USD: a log-return loss v% is 100 000 (1 - exp(-v/100))
vb = N['var']['btc']
V.put('vx.e', math.exp(-vb['hs_v'] / 100), 4)
V.put('vx.z', stats.norm.ppf(0.01), 3)
V.put('vx.phi', stats.norm.pdf(stats.norm.ppf(0.025)), 4)
# Hill: the k-th moment is finite only for k < alpha
V.put('hl.btc', s['btc']['hill_left'], 2)
for k in ('btc', 'eth', 'sol'):
    V.put(f'ratio.{k}', s[k]['ann_vol'] / s['sp500']['ann_vol'], 1)
V.raw('dd.d17', str(ep[1]['days_under']))
V.raw('dd.d21', str(ep[3]['days_under']))

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's Question and Route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: do the statistical tools of Chapters 2--10 work for a market that never closes, has no cash flows and can lose 80\\% of its value?',
       '\\textbf{Întrebarea}: funcționează instrumentele statistice din Capitolele 2--10 pe o piață care nu se închide niciodată, nu are fluxuri de numerar și poate pierde 80\\% din valoare?'),
     [T('the answer is a statistics-of-markets view of crypto assets, not a course in cryptography', 'răspunsul este o privire statistică asupra activelor cripto, nu un curs de criptografie')]),
    (T('\\textbf{Route} of the chapter', '\\textbf{Traseul} capitolului'),
     [T('Bitcoin and Ethereum: blockchain, proof of work and of stake, the halving', 'Bitcoin și Ethereum: blockchain, proof of work și proof of stake, halving-ul'),
      T('crypto as an asset class: annualisation with 365 days, tails, volatility clustering, GARCH, efficiency, the weekend', 'cripto ca o clasă de active: anualizarea cu 365 de zile, cozile, volatility clustering, GARCH, eficiența, weekendul'),
      T('correlation with equities and gold; bubbles and drawdowns; the spot Bitcoin ETFs of 2024', 'corelația cu acțiunile și aurul; bulele și drawdown-urile; ETF-urile spot pe Bitcoin din 2024'),
      T('stablecoins: types, supply, peg deviations, USDC in March 2023, the collapse of TerraUSD; regulation; VaR 1\\% and ES 2.5\\%', 'stablecoins: tipuri, ofertă, abateri de la paritate, USDC în martie 2023, prăbușirea TerraUSD; reglementarea; VaR 1\\% și ES 2,5\\%')])))

D.frame(T('Learning Outcomes', 'Rezultatele învățării'), items(
    T('Explain how a blockchain records ownership, why mining costs energy and how the Bitcoin supply follows from the halving rule',
      'Explicați cum înregistrează un blockchain proprietatea, de ce mining-ul consumă energie și cum rezultă oferta de Bitcoin din regula halving-ului'),
    T('Annualise crypto returns and volatilities with the actual frequency (365 days) and say what goes wrong with 252',
      'Anualizați randamentele și volatilitățile cripto cu frecvența reală (365 de zile) și explicați eroarea introdusă de folosirea a 252 de zile'),
    T('Estimate the tail index, a GARCH(1,1)-t model and variance-ratio tests for Bitcoin and compare them with the S\\&P 500',
      'Estimați tail index-ul, un model GARCH(1,1)-t și testele raportului varianțelor pentru Bitcoin și comparați-le cu S\\&P 500'),
    T('Measure the correlation of crypto with equities and gold over time, and the drawdowns of a crypto asset',
      'Măsurați în timp corelația activelor cripto cu acțiunile și aurul, precum și drawdown-urile unui activ cripto'),
    T('Describe the three types of stablecoins, measure a peg deviation in basis points and read a depeg episode',
      'Descrieți cele trei tipuri de stablecoins, măsurați abaterea de la paritate în puncte de bază și interpretați un episod de depeg'),
    T('Compute VaR 1\\% and ES 2.5\\% of a crypto position and backtest them', 'Calculați VaR 1\\% și ES 2,5\\% ale unei poziții cripto și verificați-le prin backtesting')))

D.frame(T('Reading and Tools', 'Bibliografie și instrumente'), items(
    (T('Textbook: \\refFHH, Ch.~23 (cryptocurrencies)', 'Manual: \\refFHH, cap.~23 (criptomonede)'),
     [T('surveys: \\refHHR; crypto index CRIX: \\refTH', 'sinteze: \\refHHR; indicele cripto CRIX: \\refTH'),
      T('risks and returns: \\refLT; stablecoins: \\refGorton, \\refLVN', 'riscuri și randamente: \\refLT; stablecoins: \\refGorton, \\refLVN')]),
    (T('Python Quantlets of this chapter: \\href{' + QLURL + '}{Quantlets/Ch\\_14}', 'Quantlet-urile Python ale capitolului: \\href{' + QLURL + '}{Quantlets/Ch\\_14}'),
     [T('daily data from EODHD (crypto close, 7 days a week); stablecoin supply from \\refDL', 'date zilnice de la EODHD (închiderea cripto, 7 zile pe săptămînă); oferta de stablecoins de la \\refDL')]),
    T('Lecture notebook: \\href{\\colaburl{notebooks/EN/chapter14_lecture_notebook.ipynb}}{open in Google Colab}',
      'Notebook-ul cursului: \\href{\\colaburl{notebooks/EN/chapter14_lecture_notebook.ipynb}}{deschideți în Google Colab}'),
    T('Video courses: \\quantinar{Introduction to Blockchain and Cryptocurrencies}{https://quantinar.com/course/134/introduction-to-blockchain-and-cryptocurrencies}; \\quantinar{Cryptocurrency as an Asset Class}{https://quantinar.com/course/55/cryptoasset}',
      'Cursuri video: \\quantinar{Introduction to Blockchain and Cryptocurrencies}{https://quantinar.com/course/134/introduction-to-blockchain-and-cryptocurrencies}; \\quantinar{Cryptocurrency as an Asset Class}{https://quantinar.com/course/55/cryptoasset}')))

# =============================================================================
# 1. BITCOIN ȘI ETHEREUM
# =============================================================================
D.section('Bitcoin and Ethereum in Brief', 'Bitcoin și Ethereum pe scurt')

D.frame(T('From a White Paper to a Market', 'De la o lucrare de nouă pagini la o piață'), cols(items(
    (T('31 October 2008: \\refNakamoto proposes ``a peer-to-peer electronic cash system\'\'', '31 octombrie 2008: \\refNakamoto propune „un sistem de numerar electronic peer-to-peer”'),
     [T('\\textbf{peer-to-peer}: users pay each other directly, without a bank in the middle', '\\textbf{peer-to-peer}: utilizatorii își plătesc direct unii altora, fără o bancă la mijloc'),
      T('the problem solved: \\textbf{double spending}, i.e.\\ paying twice with the same digital coin', 'problema rezolvată: \\textbf{dubla cheltuire} (double spending), adică plata de două ori cu aceeași monedă digitală')]),
    (T('3 January 2009: the first block; 2010: the first trades on exchanges', '3 ianuarie 2009: primul bloc; 2010: primele tranzacții pe exchange-uri'),
     [T('\\textbf{exchange}: a trading platform where crypto assets are bought and sold against dollars or other crypto assets', '\\textbf{exchange}: o platformă de tranzacționare unde activele cripto se cumpără și se vînd contra dolari sau alte active cripto')]),
    (T('\\textbf{Crypto asset}: a digital asset whose ownership is recorded on a blockchain', '\\textbf{Activ cripto}: un activ digital a cărui proprietate este înregistrată într-un blockchain'),
     [T('the author, ``Satoshi Nakamoto\'\', is still unknown; the coins mined in 2009--2010 have never moved', 'autorul, „Satoshi Nakamoto”, este încă necunoscut; monedele obținute prin mining în 2009--2010 nu au fost mutate niciodată')])),
    ph('paper', T('The Bitcoin white paper, 2008', 'Lucrarea care a lansat Bitcoin, 2008'), h='0.50\\textheight'), wl='0.62', wr='0.35'), 'footnotesize')

D.frame(T('The Blockchain: a Public Ledger', 'Blockchain-ul: un registru public'), items(
    (T('\\textbf{Ledger}: the list of all transactions; whoever holds the ledger knows who owns what', '\\textbf{Registru} (ledger): lista tuturor tranzacțiilor; cine ține registrul știe cine ce deține'),
     [T('banks keep private ledgers; Bitcoin keeps one public ledger, copied on thousands of computers (\\textbf{nodes})', 'băncile țin registre private; Bitcoin ține un singur registru public, copiat pe mii de calculatoare (\\textbf{noduri})')]),
    (T('\\textbf{Block}: a batch of transactions; \\textbf{hash}: a short fingerprint of a block (the function SHA-256)', '\\textbf{Bloc}: un pachet de tranzacții; \\textbf{hash}: o amprentă scurtă a unui bloc (funcția SHA-256)'),
     [T('changing one character of a block changes its hash completely', 'schimbarea unui singur caracter dintr-un bloc îi schimbă complet hash-ul'),
      T('each block contains the hash of the previous one: the blocks form a \\textbf{chain}', 'fiecare bloc conține hash-ul blocului anterior: blocurile formează un \\textbf{lanț}')]),
    (T('Consequence: rewriting an old transaction means rewriting every later block', 'Consecința: rescrierea unei tranzacții vechi înseamnă rescrierea tuturor blocurilor ulterioare'),
     [T('who may add the next block, and at what cost, is decided by the \\textbf{consensus} rule (next slide)', 'cine poate adăuga următorul bloc și cu ce cost decide regula de \\textbf{consens} (slide-ul următor)')]),
    T('Ownership: a coin belongs to whoever holds the private key of its address; a lost key is a lost coin', 'Proprietatea: o monedă aparține celui care deține cheia privată a adresei ei; o cheie pierdută înseamnă o monedă pierdută')))

D.frame(T('Proof of Work and Mining', 'Proof of work și mining-ul'), cols(items(
    (T('\\textbf{Proof of work} (PoW): to add a block, a \\textbf{miner} must find a number that makes the hash of the block smaller than a target', '\\textbf{Proof of work} (PoW): pentru a adăuga un bloc, un \\textbf{miner} trebuie să găsească un număr care face hash-ul blocului mai mic decît o țintă'),
     [T('the only method is trial and error: billions of hashes per second, on specialised machines', 'singura metodă este încercarea repetată: miliarde de hash-uri pe secundă, pe mașini specializate'),
      T('the target is adjusted every 2016 blocks so that a block arrives about every 10 minutes', 'ținta se ajustează la fiecare 2016 blocuri, astfel încît un bloc să apară cam la 10 minute')]),
    (T('The winner receives the \\textbf{block subsidy} (new coins) plus the fees of the transactions', 'Cîștigătorul primește \\textbf{subvenția blocului} (monede noi) plus comisioanele tranzacțiilor'),
     [T('to rewrite history, an attacker would need more computing power than all honest miners together', 'pentru a rescrie istoria, un atacator ar avea nevoie de mai multă putere de calcul decît toți minerii onești la un loc')]),
    T('The cost of security is electricity: this is why Bitcoin is criticised for its energy use', 'Costul securității este electricitatea: de aceea Bitcoin este criticat pentru consumul de energie')),
    ph('mining', T('A Bitcoin mining farm, 2014', 'O fermă de mining pentru Bitcoin, 2014'), h='0.36\\textheight'), wl='0.58', wr='0.39'), 'footnotesize')

D.frame(T('The Supply Rule and the Halving', 'Regula ofertei și halving-ul'), items(
    (T('Block subsidy: 50 BTC from 2009, \\textbf{halved} every 210\\,000 blocks (about four years)', 'Subvenția blocului: 50 BTC din 2009, \\textbf{înjumătățită} la fiecare 210\\,000 de blocuri (cam patru ani)'),
     [T('2012: 25; 2016: 12.5; 2020: 6.25; April 2024: 3.125 BTC per block', '2012: 25; 2016: 12,5; 2020: 6,25; aprilie 2024: 3,125 BTC pe bloc'),
      T('the event of halving the subsidy is called the \\textbf{halving}', 'evenimentul înjumătățirii subvenției se numește \\textbf{halving}')]),
    (T('Total supply: a geometric series', 'Oferta totală: o serie geometrică'),
     [T('$210\\,000 \\times 50 \\times (1 + \\tfrac12 + \\tfrac14 + \\dots) = 210\\,000 \\times 50 \\times 2 = 21$ million BTC', '$210\\,000 \\times 50 \\times (1 + \\tfrac12 + \\tfrac14 + \\dots) = 210\\,000 \\times 50 \\times 2 = 21$ de milioane de BTC'),
      T('about @{hv.s24} million at the start of 2024, @{hv.s30} million in 2030: almost all coins already exist', 'circa @{hv.s24} milioane la începutul lui 2024, @{hv.s30} milioane în 2030: aproape toate monedele există deja')]),
    (T('The halving dates are known years in advance', 'Datele halving-urilor se cunosc cu ani înainte'),
     [T('in an efficient market (Chapter 7), a predictable cut in new supply should already be in the price', 'pe o piață eficientă (Capitolul 7), o reducere previzibilă a ofertei noi ar trebui să fie deja inclusă în preț')])))

chart(T('Bitcoin Price and Supply', 'Prețul și oferta de Bitcoin'), 'sfm_ch14_halving', 'SFM_ch14_bitcoin_basics', [
    T('Left: Bitcoin close in USD (log scale), @{hv.start} USD in September 2014 and @{hv.end} USD on @{end}; dashed: the halvings of 2016, 2020 and 2024. Right: supply from the protocol rule',
      'Stînga: închiderea Bitcoin în USD (scală logaritmică), @{hv.start} USD în septembrie 2014 și @{hv.end} USD pe @{end}; linii întrerupte: halving-urile din 2016, 2020 și 2024. Dreapta: oferta din regula protocolului'),
    T('Interpretation: one year after the halvings the price was @{hv.2016.ch}\\%, @{hv.2020.ch}\\% and @{hv.2024.ch}\\% higher; three observations are not evidence, and the effect shrinks',
      'Interpretare: la un an după halving-uri, prețul era mai mare cu @{hv.2016.ch}\\%, @{hv.2020.ch}\\% și @{hv.2024.ch}\\%; trei observații nu sînt o dovadă, iar efectul se micșorează')],
    h='0.50\\textheight')

D.frame(T('An Anonymous Founder', 'Un fondator anonim'), cols(items(
    (T('Nobody knows who ``Satoshi Nakamoto\'\' is; the founder stopped writing in 2011', 'Nimeni nu știe cine este „Satoshi Nakamoto”; fondatorul a încetat să mai scrie în 2011'),
     [T('the rules of Bitcoin are changed only if most nodes accept the new software', 'regulile Bitcoin se schimbă doar dacă majoritatea nodurilor acceptă noul software')]),
    (T('Consequences for a statistician', 'Consecințe pentru un statistician'),
     [T('no company, no profits, no dividends: no cash flows to discount', 'nicio companie, niciun profit, niciun dividend: niciun flux de numerar de actualizat'),
      T('the price reflects only demand: payments, speculation, belief in scarcity \\refBHL', 'prețul reflectă doar cererea: plăți, speculație, încrederea în raritate \\refBHL'),
      T('so the usual valuation models do not apply; we study the price as a statistical object', 'deci modelele obișnuite de evaluare nu se aplică; studiem prețul ca obiect statistic')]),
    T('Trades happen around the clock on hundreds of exchanges, 365 days a year', 'Tranzacțiile au loc non-stop pe sute de exchange-uri, 365 de zile pe an')),
    ph('bust', T('A bust of Satoshi Nakamoto, Budapest', 'Bustul lui Satoshi Nakamoto, Budapesta'), h='0.46\\textheight'), wl='0.62', wr='0.35'), 'footnotesize')

D.frame(T('Ethereum, Smart Contracts and Proof of Stake', 'Ethereum, contractele inteligente și proof of stake'), cols(items(
    (T('2014: \\refButerin proposes Ethereum, a blockchain that runs programs; launched in July 2015', '2014: \\refButerin propune Ethereum, un blockchain care execută programe; lansat în iulie 2015'),
     [T('\\textbf{smart contract}: a program stored on the blockchain that executes a transaction when its conditions are met', '\\textbf{contract inteligent} (smart contract): un program stocat în blockchain care execută o tranzacție cînd condițiile lui sînt îndeplinite'),
      T('on top of it: tokens, stablecoins and DeFi (decentralised finance: lending and trading without intermediaries)', 'pe baza lui: token-uri, stablecoins și DeFi (finanțe descentralizate: credit și tranzacționare fără intermediari)')]),
    (T('\\textbf{Proof of stake} (PoS): the right to add a block is drawn in proportion to the coins locked as a deposit (\\textbf{stake})', '\\textbf{Proof of stake} (PoS): dreptul de a adăuga un bloc se trage la sorți proporțional cu monedele blocate drept garanție (\\textbf{stake})'),
     [T('cheating is punished by losing the stake; no race of computing power', 'frauda se pedepsește prin pierderea garanției; nu mai există o cursă a puterii de calcul'),
      T('15 September 2022 (``the Merge\'\'): Ethereum moves from PoW to PoS; energy use falls by more than 99.9\\% \\refMerge', '15 septembrie 2022 („the Merge”): Ethereum trece de la PoW la PoS; consumul de energie scade cu peste 99,9\\% \\refMerge')])),
    ph('buterin', T('Vitalik Buterin, 2015', 'Vitalik Buterin, 2015'), h='0.42\\textheight'), wl='0.64', wr='0.33'), 'footnotesize')

D.frame(T('Prices and Data', 'Prețuri și date'), cols(items(
    (T('There is no single exchange: each platform has its own price; prices differ across countries \\refMS', 'Nu există un exchange unic: fiecare platformă are propriul preț; prețurile diferă între țări \\refMS'),
     [T('a data provider aggregates the platforms into one daily series; here: EODHD, close at midnight UTC (Coordinated Universal Time)', 'un furnizor de date agregă platformele într-o serie zilnică; aici: EODHD, închiderea la miezul nopții UTC (Coordinated Universal Time, ora universală coordonată)')]),
    (T('Seven trading days a week: no weekend gaps, no holidays', 'Șapte zile de tranzacționare pe săptămînă: fără goluri de weekend, fără sărbători'),
     [T('Bitcoin: @{s.btc.n} daily returns since @{s.btc.y0}; the S\\&P 500: @{s.sp500.n} returns in the same period', 'Bitcoin: @{s.btc.n} randamente zilnice din @{s.btc.y0}; S\\&P 500: @{s.sp500.n} de randamente în aceeași perioadă')]),
    (T('Market indices of many coins exist, e.g.\\ CRIX (Crypto Index) \\refTH', 'Există indici de piață cu multe monede, de exemplu CRIX (Crypto Index) \\refTH'),
     [T('CRIX chooses the number of coins with the AIC criterion (Chapter 6): few coins describe the whole market', 'CRIX alege numărul de monede cu criteriul AIC (Capitolul 6): puține monede descriu întreaga piață')]),
    T('Returns: $r_t = 100(\\ln P_t - \\ln P_{t-1})$ in \\%, each series on its own calendar', 'Randamentele: $r_t = 100(\\ln P_t - \\ln P_{t-1})$ în \\%, fiecare serie pe propriul calendar')),
    ph('atm', T('A Bitcoin ATM, Prague', 'Un bancomat Bitcoin, Praga'), h='0.40\\textheight'), wl='0.64', wr='0.33'), 'footnotesize')

D.recap(('Bitcoin and Ethereum in Brief', 'noțiuni de bază despre Bitcoin și Ethereum'), [
    T('A blockchain is a public chain of blocks linked by hashes; changing the past means redoing all later work', 'Un blockchain este un lanț public de blocuri legate prin hash-uri; schimbarea trecutului înseamnă refacerea întregii munci ulterioare'),
    T('PoW pays miners for computing; PoS asks validators for a deposit; Ethereum switched in 2022', 'PoW plătește minerii pentru calcul; PoS cere validatorilor o garanție; Ethereum a trecut la PoS în 2022'),
    T('The Bitcoin supply is fixed by the halving rule at 21 million; there are no cash flows, only prices', 'Oferta de Bitcoin este fixată de regula halving-ului la 21 de milioane; nu există fluxuri de numerar, doar prețuri')])

# =============================================================================
# 2. CRIPTO CA O CLASĂ DE ACTIVE
# =============================================================================
D.section('Crypto as an Asset Class', 'Cripto ca o clasă de active')

D.frame(T('Annualisation with the Actual Frequency (1/2)', 'Anualizarea cu frecvența reală (1/2)'), items(
    (T('With $A$ observations a year and i.i.d.\\ daily returns (Chapter 1):', 'Cu $A$ observații pe an și randamente zilnice i.i.d. (Capitolul 1):'),
     ['$\\text{' + T('annual mean', 'media anuală') + '} = A\\,\\bar r, \\qquad \\text{' + T('annual volatility', 'volatilitatea anuală') + '} = \\sqrt{A}\\,s$',
      T('$\\bar r$: the mean daily return; $s$: the standard deviation of daily returns; $A$: the number of returns per year', '$\\bar r$: randamentul zilnic mediu; $s$: abaterea standard a randamentelor zilnice; $A$: numărul de randamente pe an'),
      T('i.i.d.: independent and identically distributed; then the variances of the $A$ daily returns add up, hence $\\sqrt{A}$', 'i.i.d.: independente și identic distribuite; atunci varianțele celor $A$ randamente zilnice se adună, de aici $\\sqrt{A}$')]),
    (T('The right $A$ is the actual number of observations per year', 'Valoarea corectă a lui $A$ este numărul real de observații pe an'),
     [T('equities: $A \\approx 252$ trading days; crypto: $A = 365$', 'acțiunile: $A \\approx 252$ de zile de tranzacționare; cripto: $A = 365$'),
      T('actual frequency in our data: $A = @{s.btc.ppy}$ for Bitcoin, @{s.sp500.ppy} for the S\\&P 500', 'frecvența reală în datele noastre: $A = @{s.btc.ppy}$ pentru Bitcoin, @{s.sp500.ppy} pentru S\\&P 500')])))

D.frame(T('Annualisation with the Actual Frequency (2/2)', 'Anualizarea cu frecvența reală (2/2)'), items(
    (T('\\textbf{Worked example}: Bitcoin, $\\bar r = @{s.btc.mean}\\%$, $s = @{s.btc.sd}\\%$ per day', '\\textbf{Exemplu lucrat}: Bitcoin, $\\bar r = @{s.btc.mean}\\%$, $s = @{s.btc.sd}\\%$ pe zi'),
     [T('right: $\\sqrt{365} \\times @{s.btc.sd} = @{an.s365} \\times @{s.btc.sd} = @{s.btc.av}\\%$; mean $365 \\times @{s.btc.mean} = @{s.btc.am}\\%$', 'corect: $\\sqrt{365} \\times @{s.btc.sd} = @{an.s365} \\times @{s.btc.sd} = @{s.btc.av}\\%$; media $365 \\times @{s.btc.mean} = @{s.btc.am}\\%$'),
      T('wrong: $\\sqrt{252} \\times @{s.btc.sd} = @{s.btc.av252}\\%$ and $252 \\times @{s.btc.mean} = @{s.btc.am252}\\%$', 'greșit: $\\sqrt{252} \\times @{s.btc.sd} = @{s.btc.av252}\\%$ și $252 \\times @{s.btc.mean} = @{s.btc.am252}\\%$')]),
    (T('The 252 rule understates the volatility by @{an.under}\\% and the mean by @{an.underm}\\%', 'Regula cu 252 subestimează volatilitatea cu @{an.under}\\% și media cu @{an.underm}\\%'),
     [T('mixing the two (mean $\\times 252$, volatility $\\times\\sqrt{365}$) also biases the Sharpe ratio', 'amestecul lor (media $\\times 252$, volatilitatea $\\times\\sqrt{365}$) deformează și raportul Sharpe'),
      T('comparisons with equities: annualise each series with its own $A$, never with a common one', 'comparațiile cu acțiunile: anualizăm fiecare serie cu propriul $A$, niciodată cu unul comun')])))

TB = '>{\\raggedright\\arraybackslash}'
ASSETS = ['btc', 'eth', 'sol', 'sp500', 'gold']
NM = {'btc': 'Bitcoin', 'eth': 'Ethereum', 'sol': 'Solana', 'sp500': 'S\\&P 500', 'gold': T('Gold', 'Aur')}
D.frame(T('Five Assets Side by Side', 'Cinci active comparate'), table(
    'lrrrrrrrr', T('& since & $n$ & mean (\\%) & sd (\\%) & ann.\\ vol.\\ (\\%) & skew. & exc.\\ kurt. & min (\\%)',
                   '& din & $n$ & media (\\%) & ab.\\ std.\\ (\\%) & vol.\\ anuală (\\%) & asim. & exces de boltire & min (\\%)'),
    [f'{NM[k]} & @{{s.{k}.y0}} & @{{s.{k}.n}} & $@{{s.{k}.mean}}$ & $@{{s.{k}.sd}}$ & $@{{s.{k}.av}}$ & $@{{s.{k}.skew}}$ & $@{{s.{k}.k}}$ & $@{{s.{k}.min}}$' for k in ASSETS],
    size='scriptsize') + items(
    T('Daily log returns until @{end}; annual volatility $= \\sqrt{A}\\,s$ with the actual frequency $A$ of each series; gold: XAU/USD', 'Randamente logaritmice zilnice pînă pe @{end}; volatilitatea anuală $= \\sqrt{A}\\,s$ cu frecvența reală $A$ a fiecărei serii; aurul: XAU/USD'),
    (T('Interpretation', 'Interpretare'),
     [T('Bitcoin is @{ratio.btc} times as volatile as the S\\&P 500 (@{s.btc.av}\\% against @{s.sp500.av}\\%); Ethereum @{ratio.eth} times, Solana @{ratio.sol} times', 'Bitcoin este de @{ratio.btc} ori mai volatil decît S\\&P 500 (@{s.btc.av}\\% față de @{s.sp500.av}\\%); Ethereum de @{ratio.eth} ori, Solana de @{ratio.sol} ori'),
      T('on @{s.btc.sh5}\\% of the days Bitcoin moves by more than 5\\%; the worst day: $@{s.btc.min}\\%$ on @{s.btc.mind}', 'în @{s.btc.sh5}\\% din zile, Bitcoin se mișcă cu peste 5\\%; cea mai proastă zi: $@{s.btc.min}\\%$ pe @{s.btc.mind}'),
      T('all five have negative skewness and excess kurtosis: the stylised facts of Chapter 2 hold, in a stronger form', 'toate cinci au asimetrie negativă și exces de boltire: faptele stilizate din Capitolul 2 se păstrează, într-o formă mai puternică')])), 'footnotesize')

chart(T('Volatility Through Time', 'Volatilitatea în timp'), 'sfm_ch14_rolling_vol', 'SFM_ch14_asset_class', [
    T('Volatility on about one month: 30 daily returns $\\times\\sqrt{365}$ for crypto, 21 returns $\\times\\sqrt{252}$ for the S\\&P 500', 'Volatilitatea pe circa o lună: 30 de randamente zilnice $\\times\\sqrt{365}$ pentru cripto, 21 de randamente $\\times\\sqrt{252}$ pentru S\\&P 500'),
    T('Interpretation: medians @{rv.btc.med}\\% (Bitcoin), @{rv.eth.med}\\% (Ethereum), @{rv.sp500.med}\\% (S\\&P 500); the median ratio Bitcoin/S\\&P 500 is @{rv.ratio}; all three peak in March--April 2020',
      'Interpretare: medianele @{rv.btc.med}\\% (Bitcoin), @{rv.eth.med}\\% (Ethereum), @{rv.sp500.med}\\% (S\\&P 500); raportul median Bitcoin/S\\&P 500 este @{rv.ratio}; toate trei ating maximul în martie--aprilie 2020'),
    T('Bitcoin\'s volatility trends down, but it was below that of the S\\&P 500 on only @{rv.below}\\% of the days', 'Volatilitatea Bitcoin scade în timp, dar a fost sub cea a S\\&P 500 doar în @{rv.below}\\% din zile')],
    h='0.46\\textheight')

D.frame(T('Case Study: Risks and Returns of Cryptocurrency', 'Studiu de caz: riscurile și randamentele criptomonedelor'), items(
    (T('\\refLT: Bitcoin, Ripple and Ethereum, daily and weekly returns, 2011--2018', '\\refLT: Bitcoin, Ripple și Ethereum, randamente zilnice și săptămînale, 2011--2018'),
     [T('design: regress crypto returns on stock-market factors, currencies, commodities and macroeconomic factors', 'metoda: regresia randamentelor cripto pe factorii pieței de acțiuni, pe valute, mărfuri și factori macroeconomici')]),
    (T('Findings', 'Rezultatele'),
     [T('crypto returns have almost no exposure to the usual factors: a separate asset class', 'randamentele cripto nu au aproape nicio expunere la factorii obișnuiți: o clasă de active separată'),
      T('strong time-series momentum: high returns this week predict high returns in the next weeks', 'momentum puternic în timp: randamentele mari dintr-o săptămînă prognozează randamente mari în săptămînile următoare'),
      T('investor attention (Google searches, Twitter posts) predicts returns', 'atenția investitorilor (căutări Google, postări pe Twitter) prognozează randamentele')]),
    (T('Follow-up \\refLTW: three crypto factors (market, size, momentum) price the cross-section of coins', 'Continuare \\refLTW: trei factori cripto (piața, dimensiunea, momentum) explică randamentele transversale ale monedelor'),
     [T('the link with equities has changed since 2020 (Section 5): results depend on the sample period', 'legătura cu acțiunile s-a schimbat după 2020 (secțiunea 5): rezultatele depind de perioada eșantionului')])))

D.frame(T('Is the Weekend Different?', 'Este weekendul diferit?'), items(
    (T('Stock markets close at weekends; crypto markets do not, but banks, institutional traders and news slow down', 'Bursele de acțiuni se închid în weekend; piețele cripto nu, dar băncile, investitorii instituționali și știrile încetinesc'),
     [T('\\textbf{weekday effect}: different mean or variance of returns on some days of the week \\refCP', '\\textbf{efectul zilei din săptămînă}: medie sau varianță diferită a randamentelor în anumite zile \\refCP')]),
    (T('Two tests on Bitcoin, each day labelled by its closing date', 'Două teste pe Bitcoin, fiecare zi etichetată după data închiderii'),
     [T('mean: $r_t = a + b\\,W_t + \\varepsilon_t$, $W_t = 1$ on Saturday and Sunday, 0 otherwise; HAC (Newey--West) standard errors', 'media: $r_t = a + b\\,W_t + \\varepsilon_t$, cu $W_t = 1$ sîmbăta și duminica și 0 în rest; erori standard HAC (Newey--West)'),
      T('$a$: the mean return on weekdays; $b$: the weekend minus weekday difference; $b = 0$: no weekend effect in the mean', '$a$: randamentul mediu în zilele lucrătoare; $b$: diferența weekend minus zile lucrătoare; $b = 0$: niciun efect de weekend în medie'),
      T('variance: the Brown--Forsythe test (Levene\'s test around the median), robust to heavy tails', 'varianța: testul Brown--Forsythe (testul Levene în jurul medianei), robust la cozi groase')]),
    T('Ratio of the weekend to the weekday variance: equal to 1 if every day had the same risk', 'Raportul dintre varianța din weekend și cea din zilele lucrătoare: egal cu 1 dacă fiecare zi ar avea același risc')))

chart(T('Bitcoin Volatility by Day of the Week', 'Volatilitatea Bitcoin pe zile ale săptămînii'), 'sfm_ch14_weekday', 'SFM_ch14_asset_class', [
    T('Standard deviation of daily returns by closing day; three periods', 'Abaterea standard a randamentelor zilnice după ziua închiderii; trei perioade'),
    T('Weekend dummy $b$ (p-value): $@{wd.2014.b}$ (@{wd.2014.p}), $@{wd.2020.b}$ (@{wd.2020.p}), $@{wd.2024.b}$ (@{wd.2024.p}): no weekend effect in the mean',
      'Coeficientul variabilei weekend, $b$ (p-value): $@{wd.2014.b}$ (@{wd.2014.p}); $@{wd.2020.b}$ (@{wd.2020.p}); $@{wd.2024.b}$ (@{wd.2024.p}): niciun efect de weekend în medie'),
    T('Interpretation: weekend/weekday variance @{wd.2014.vr}, @{wd.2020.vr}, @{wd.2024.vr} (Brown--Forsythe p @{wd.2014.pbf}, @{wd.2020.pbf}, @{wd.2024.pbf}): crypto is calmer when traditional markets close',
      'Interpretare: varianța weekend/zile lucrătoare @{wd.2014.vr}, @{wd.2020.vr}, @{wd.2024.vr} (Brown--Forsythe p @{wd.2014.pbf}; @{wd.2020.pbf}; @{wd.2024.pbf}): piața cripto este mai liniștită cînd piețele tradiționale sînt închise')],
    h='0.44\\textheight')

D.recap(('Crypto as an Asset Class', 'cripto ca o clasă de active'), [
    T('Annualise crypto with 365 days: Bitcoin @{s.btc.av}\\% a year, not @{s.btc.av252}\\%', 'Anualizăm cripto cu 365 de zile: Bitcoin @{s.btc.av}\\% pe an, nu @{s.btc.av252}\\%'),
    T('Crypto volatility is @{ratio.btc} (Bitcoin) to @{ratio.sol} (Solana) times that of the S\\&P 500, and decreasing', 'Volatilitatea cripto este de @{ratio.btc} (Bitcoin) pînă la @{ratio.sol} (Solana) ori cea a S\\&P 500 și este în scădere'),
    T('No weekend effect in the mean, a strong one in the variance', 'Niciun efect de weekend în medie, un efect puternic în varianță')])

# =============================================================================
# 3. COZI ȘI VOLATILITY CLUSTERING
# =============================================================================
D.section('Tails and Volatility Clustering', 'Cozi și volatility clustering')

D.frame(T('Heavy Tails: the Hill Estimator (1/2)', 'Cozi groase: estimatorul Hill (1/2)'), items(
    T('\\refHill (Chapter 5): the tail index from the $k$ largest losses (or gains)', '\\refHill (Capitolul 5): tail index-ul estimat din cele mai mari $k$ pierderi (sau cîștiguri)')
    + ' \\[ \\hat\\alpha = \\Big[\\frac1k\\sum_{i=1}^k \\ln\\frac{X_{(i)}}{X_{(k+1)}}\\Big]^{-1}, \\qquad CI_{95\\%} = \\hat\\alpha\\Big(1 \\pm \\frac{' + T('1.96', '1{,}96') + '}{\\sqrt{k}}\\Big) \\]',
    (T('Notation', 'Notațiile'),
     [T('$X_{(1)} \\ge X_{(2)} \\ge \\dots$: the losses (or gains) sorted from the largest; $X_{(k+1)}$: the threshold of the tail', '$X_{(1)} \\ge X_{(2)} \\ge \\dots$: pierderile (sau cîștigurile) ordonate descrescător; $X_{(k+1)}$: pragul cozii'),
      T('$k$: the number of tail observations, here $k = 2.5\\%$ of $n$; $n$: the sample size; CI: confidence interval', '$k$: numărul observațiilor din coadă, aici $k = 2{,}5\\%$ din $n$; $n$: volumul eșantionului; CI: intervalul de încredere')]),
    (T('Meaning of $\\alpha$: $P(X > x) \\approx c\\,x^{-\\alpha}$ for large $x$ ($c$: a constant)', 'Semnificația lui $\\alpha$: $P(X > x) \\approx c\\,x^{-\\alpha}$ pentru $x$ mare ($c$: o constantă)'),
     [T('a smaller $\\alpha$ means a heavier tail', 'un $\\alpha$ mai mic înseamnă o coadă mai groasă'),
      T('moments of order $\\ge \\alpha$ are infinite: $\\alpha < 4$ means an infinite kurtosis', 'momentele de ordin $\\ge \\alpha$ sînt infinite: $\\alpha < 4$ înseamnă o boltire infinită')])))

D.frame(T('Heavy Tails: the Hill Estimator (2/2)', 'Cozi groase: estimatorul Hill (2/2)'), table(
    'lrrrr', T('& left tail $\\hat\\alpha$ & 95\\% CI & right tail $\\hat\\alpha$ & 95\\% CI', '& coada stîngă $\\hat\\alpha$ & CI 95\\% & coada dreaptă $\\hat\\alpha$ & CI 95\\%'),
    [f'{NM[k]} & $@{{s.{k}.hill_left}}$ & $[@{{s.{k}.hill_left_lo}}, @{{s.{k}.hill_left_hi}}]$ & $@{{s.{k}.hill_right}}$ & $[@{{s.{k}.hill_right_lo}}, @{{s.{k}.hill_right_hi}}]$' for k in ASSETS],
    size='scriptsize') + items(
    (T('Interpretation', 'Interpretare'),
     [T('Bitcoin\'s left tail ($\\hat\\alpha = @{hl.btc}$) is heavier than its right tail', 'coada stîngă a Bitcoin ($\\hat\\alpha = @{hl.btc}$) este mai groasă decît coada dreaptă'),
      T('with $\\alpha < 4$ the kurtosis is not finite, so the sample kurtosis is not a stable number', 'cu $\\alpha < 4$, boltirea nu este finită, deci boltirea de selecție nu este o valoare stabilă'),
      T('the intervals of crypto and equities overlap', 'intervalele pentru cripto și acțiuni se suprapun')])), 'footnotesize')

D.frame(T('Reading the ACF and the Ljung--Box Test', 'Interpretarea ACF și a testului Ljung--Box'), items(
    (T('\\textbf{ACF} (autocorrelation function): $\\hat\\rho(k)$, the sample correlation between $r_t$ and $r_{t-k}$; $k$: the lag in days', '\\textbf{ACF} (funcția de autocorelație): $\\hat\\rho(k)$, corelația de selecție dintre $r_t$ și $r_{t-k}$; $k$: lagul, în zile'),
     [T('under independence, $\\hat\\rho(k)$ lies within $\\pm 1.96/\\sqrt{n}$ with probability 95\\% ($n$: the number of returns)', 'în ipoteza de independență, $\\hat\\rho(k)$ se află în intervalul $\\pm 1{,}96/\\sqrt{n}$ cu probabilitatea 95\\% ($n$: numărul de randamente)'),
      T('ACF of $r_t$: is the direction predictable? ACF of $r_t^2$: is the size of the move predictable (volatility clustering)?', 'ACF a lui $r_t$: este direcția previzibilă? ACF a lui $r_t^2$: este mărimea mișcării previzibilă (volatility clustering)?')]),
    T('\\textbf{Ljung--Box} test of the first 10 autocorrelations together:', 'Testul \\textbf{Ljung--Box} pentru primele 10 autocorelații împreună:')
    + ' \\[ Q(10) = n(n + 2)\\sum_{k=1}^{10}\\frac{\\hat\\rho(k)^2}{n - k} \\]',
    (T('Reading', 'Interpretarea'),
     [T('without autocorrelation, $Q(10)$ follows a $\\chi^2$ distribution with 10 degrees of freedom', 'în absența autocorelației, $Q(10)$ urmează o distribuție $\\chi^2$ cu 10 grade de libertate'),
      T('a large $Q(10)$ and a small p-value: the autocorrelations are jointly significant', 'un $Q(10)$ mare și un p-value mic: autocorelațiile sînt semnificative împreună')])))

chart(T('Volatility Clustering in Bitcoin', 'Volatility clustering la Bitcoin'), 'sfm_ch14_acf', 'SFM_ch14_garch_efficiency', [
    T('ACF (autocorrelation function) of $r_t$ and of $r_t^2$, lags 1--30; dashed: $\\pm 1.96/\\sqrt{n} = \\pm @{acf.band}$', 'ACF (funcția de autocorelație) a lui $r_t$ și a lui $r_t^2$, lagurile 1--30; linii întrerupte: $\\pm 1{,}96/\\sqrt{n} = \\pm @{acf.band}$'),
    T('Returns: $\\hat\\rho(1) = @{acf.rho1}$, @{acf.nout} of 30 lags outside the band; squared returns: $\\hat\\rho(1) = @{acf.rho1sq}$, @{acf.noutsq} of 30 outside',
      'Randamentele: $\\hat\\rho(1) = @{acf.rho1}$, @{acf.nout} din 30 de laguri în afara benzii; pătratele: $\\hat\\rho(1) = @{acf.rho1sq}$, @{acf.noutsq} din 30 în afara benzii'),
    T('Interpretation: Ljung--Box $Q(10)$ is @{acf.lbr} for $r_t$ and @{acf.lbsq} for $r_t^2$ (p @{acf.lbsqp}): the direction is hard to predict, the size of the move is not',
      'Interpretare: statistica Ljung--Box $Q(10)$ este @{acf.lbr} pentru $r_t$ și @{acf.lbsq} pentru $r_t^2$ (p @{acf.lbsqp}): direcția este greu de prognozat, mărimea mișcării nu')],
    h='0.46\\textheight')

GA = ['btc', 'eth', 'sp500', 'gold']
D.frame(T('GARCH(1,1)-t for Crypto and Traditional Assets (1/2)', 'GARCH(1,1)-t pentru active cripto și tradiționale (1/2)'), items(
    T('\\refBoll (Chapter 9): today\'s variance is a weighted sum of a constant, yesterday\'s squared shock and yesterday\'s variance', '\\refBoll (Capitolul 9): varianța de azi este o sumă ponderată a unei constante, a pătratului șocului de ieri și a varianței de ieri')
    + ' \\[ \\sigma_t^2 = \\omega + \\alpha\\,\\varepsilon_{t-1}^2 + \\beta\\,\\sigma_{t-1}^2, \\qquad \\varepsilon_t = \\sigma_t z_t \\]',
    (T('Notation', 'Notațiile'),
     [T('$\\sigma_t^2$: the conditional variance of day $t$; $\\varepsilon_t$: the shock (the demeaned return); $z_t$: Student-t with $\\nu$ degrees of freedom, scaled to variance 1', '$\\sigma_t^2$: varianța condiționată a zilei $t$; $\\varepsilon_t$: șocul (randamentul centrat); $z_t$: Student-t cu $\\nu$ grade de libertate, scalat la varianța 1'),
      T('$\\omega > 0$: the constant; $\\alpha \\ge 0$: the reaction to news; $\\beta \\ge 0$: the memory of past variance; $\\alpha + \\beta$: the persistence', '$\\omega > 0$: constanta; $\\alpha \\ge 0$: reacția la șocuri; $\\beta \\ge 0$: memoria varianței trecute; $\\alpha + \\beta$: persistența'),
      T('half-life of a shock: $\\ln 0.5/\\ln(\\alpha + \\beta)$ days, the time in which half of its effect on the variance is gone', 'timpul de înjumătățire a unui șoc: $\\ln 0{,}5/\\ln(\\alpha + \\beta)$ zile, intervalul după care jumătate din efectul lui asupra varianței a dispărut')]),
    (T('GJR: an extra term $\\gamma\\,\\varepsilon_{t-1}^2\\mathbb{1}\\{\\varepsilon_{t-1} < 0\\}$', 'GJR: un termen suplimentar $\\gamma\\,\\varepsilon_{t-1}^2\\mathbb{1}\\{\\varepsilon_{t-1} < 0\\}$'),
     [T('$\\mathbb{1}\\{\\varepsilon_{t-1} < 0\\} = 1$ after a fall; $\\gamma > 0$: falls raise volatility more than rises, the leverage effect of equities', '$\\mathbb{1}\\{\\varepsilon_{t-1} < 0\\} = 1$ după o scădere; $\\gamma > 0$: scăderile cresc volatilitatea mai mult decît creșterile, efectul de levier al acțiunilor')])))

D.frame(T('GARCH(1,1)-t for Crypto and Traditional Assets (2/2)', 'GARCH(1,1)-t pentru active cripto și tradiționale (2/2)'), table(
    'lrrrrrrr', T('& $\\hat\\omega$ & $\\hat\\alpha$ & $\\hat\\beta$ & $\\hat\\alpha + \\hat\\beta$ & half-life (days) & $\\hat\\nu$ & GJR $\\hat\\gamma$ ($t$)',
                  '& $\\hat\\omega$ & $\\hat\\alpha$ & $\\hat\\beta$ & $\\hat\\alpha + \\hat\\beta$ & timp de înjumătățire (zile) & $\\hat\\nu$ & GJR $\\hat\\gamma$ ($t$)'),
    [f'{NM[k]} & $@{{g.{k}.omega}}$ & $@{{g.{k}.alpha}}$ & $@{{g.{k}.beta}}$ & $@{{g.{k}.pers}}$ & @{{g.{k}.hl}} & $@{{g.{k}.nu}}$ & $@{{g.{k}.gamma}}$ ($@{{g.{k}.tg}}$)' for k in GA],
    size='scriptsize') + items(
    T('Estimates by maximum likelihood on daily log returns; half-life in days; $t$: the $t$-statistic of $\\hat\\gamma$', 'Estimații prin verosimilitate maximă pe randamente logaritmice zilnice; timpul de înjumătățire în zile; $t$: statistica $t$ a lui $\\hat\\gamma$')), 'footnotesize')

D.frame(T('Interpretation of the GARCH Estimates', 'Interpretarea estimărilor GARCH'), items(
    (T('Crypto: $\\hat\\alpha + \\hat\\beta = 1$ at the boundary: an integrated GARCH (IGARCH; the EWMA of Chapter 8 is the case $\\omega = 0$)', 'Cripto: $\\hat\\alpha + \\hat\\beta = 1$, la limita domeniului: un GARCH integrat (IGARCH; EWMA din Capitolul 8 este cazul $\\omega = 0$)'),
     [T('a shock to volatility does not die out within the sample; there is no finite long-run variance to which volatility returns', 'un șoc al volatilității nu se stinge în eșantion; nu există o varianță de termen lung finită spre care să revină volatilitatea'),
      T('S\\&P 500 and gold: half-lives of @{g.sp500.hl} and @{g.gold.hl} days', 'S\\&P 500 și aurul: timpi de înjumătățire de @{g.sp500.hl} și @{g.gold.hl} zile')]),
    (T('$\\hat\\nu \\approx @{g.btc.nu}$ for Bitcoin: even after GARCH filtering the tails are very heavy', '$\\hat\\nu \\approx @{g.btc.nu}$ pentru Bitcoin: chiar după filtrarea GARCH, cozile sînt foarte groase'),
     [T('the S\\&P 500 has $\\hat\\nu = @{g.sp500.nu}$', 'S\\&P 500 are $\\hat\\nu = @{g.sp500.nu}$')]),
    (T('No leverage effect in crypto: $\\hat\\gamma = @{g.btc.gamma}$ ($t = @{g.btc.tg}$) for Bitcoin, against $@{g.sp500.gamma}$ ($t = @{g.sp500.tg}$) for the S\\&P 500', 'Niciun efect de levier la cripto: $\\hat\\gamma = @{g.btc.gamma}$ ($t = @{g.btc.tg}$) pentru Bitcoin, față de $@{g.sp500.gamma}$ ($t = @{g.sp500.tg}$) pentru S\\&P 500'),
     [T('gold has $\\hat\\gamma < 0$ ($t = @{g.gold.tg}$): rises in its price raise volatility more than falls, as for a safe haven', 'aurul are $\\hat\\gamma < 0$ ($t = @{g.gold.tg}$): creșterile prețului cresc volatilitatea mai mult decît scăderile, ca la un activ de refugiu')]),
    T('\\refKatsiampa compares GARCH-type models for Bitcoin: a model with a short-run and a long-run volatility component fits best', '\\refKatsiampa compară modele de tip GARCH pentru Bitcoin: cel mai bine se potrivește un model cu o componentă de termen scurt și una de termen lung a volatilității')))

chart(T('Conditional Volatility: Bitcoin and the S\\&P 500', 'Volatilitatea condiționată: Bitcoin și S\\&P 500'), 'sfm_ch14_garch_vol', 'SFM_ch14_garch_efficiency', [
    T('GARCH(1,1)-t volatility $\\hat\\sigma_t$, annualised with $\\sqrt{365}$ (Bitcoin) and $\\sqrt{252}$ (S\\&P 500)', 'Volatilitatea GARCH(1,1)-t $\\hat\\sigma_t$, anualizată cu $\\sqrt{365}$ (Bitcoin) și $\\sqrt{252}$ (S\\&P 500)'),
    T('Interpretation: both peak in March 2020 (@{g.btc.maxs}\\% and @{g.sp500.maxs}\\%); on @{end} the model gives @{g.btc.last}\\% for Bitcoin and @{g.sp500.last}\\% for the S\\&P 500',
      'Interpretare: ambele ating maximul în martie 2020 (@{g.btc.maxs}\\% și @{g.sp500.maxs}\\%); pe @{end}, modelul dă @{g.btc.last}\\% pentru Bitcoin și @{g.sp500.last}\\% pentru S\\&P 500')],
    h='0.48\\textheight')

D.recap(('Tails and Volatility Clustering', 'cozi și volatility clustering'), [
    T('Crypto tails are heavy ($\\hat\\alpha$ between 2.5 and 3.5), as heavy in shape as those of the S\\&P 500; crypto differs in scale', 'Cozile cripto sînt groase ($\\hat\\alpha$ între 2,5 și 3,5), la fel de groase ca formă ca ale S\\&P 500; activele cripto diferă prin scală'),
    T('Volatility clusters strongly; GARCH-t fits with $\\alpha + \\beta \\approx 1$ (IGARCH)', 'Volatility clustering este puternic; GARCH-t se estimează cu $\\alpha + \\beta \\approx 1$ (IGARCH)'),
    T('No leverage effect: bad and good news move crypto volatility alike', 'Niciun efect de levier: veștile bune și cele rele mișcă la fel volatilitatea cripto')])

# =============================================================================
# 4. EFICIENȚA
# =============================================================================
D.section('Is the Crypto Market Efficient?', 'Este piața cripto eficientă?')

D.frame(T('Weak-Form Efficiency: What We Test', 'Eficiența în formă slabă: ipotezele testate'), items(
    (T('\\refFama: in a weak-form efficient market, past prices do not help to predict future returns (Chapter 7)', '\\refFama: pe o piață eficientă în formă slabă, prețurile trecute nu ajută la prognoza randamentelor viitoare (Capitolul 7)'),
     [T('forms of efficiency: weak (past prices), semi-strong (public information), strong (all information)', 'formele eficienței: slabă (prețurile trecute), semi-tare (informația publică), tare (toată informația)')]),
    (T('Tests used here', 'Testele folosite aici'),
     [T('first-order autocorrelation $\\hat\\rho(1)$ and Ljung--Box $Q(10)$', 'autocorelația de ordinul întîi $\\hat\\rho(1)$ și statistica Ljung--Box $Q(10)$'),
      T('variance ratio \\refLM: $\\mathrm{VR}(q) = \\mathrm{Var}(r_t + \\dots + r_{t-q+1})/(q\\,\\mathrm{Var}(r_t))$, equal to 1 under a random walk', 'raportul varianțelor \\refLM: $\\mathrm{VR}(q) = \\mathrm{Var}(r_t + \\dots + r_{t-q+1})/(q\\,\\mathrm{Var}(r_t))$, egal cu 1 pentru un mers aleator'),
      T('$q$: the horizon in days; $r_t + \\dots + r_{t-q+1}$: the $q$-day log return; without autocorrelation its variance is $q$ times the daily one', '$q$: orizontul, în zile; $r_t + \\dots + r_{t-q+1}$: randamentul logaritmic pe $q$ zile; fără autocorelație, varianța lui este de $q$ ori cea zilnică'),
      T('$Z^*(q)$: the version robust to volatility clustering; reject at 5\\% if $|Z^*| > 1.96$', '$Z^*(q)$: varianta robustă la volatility clustering; respingem la 5\\% dacă $|Z^*| > 1{,}96$')]),
    T('VR $> 1$: momentum (trends); VR $< 1$: mean reversion', 'VR $> 1$: momentum (tendințe); VR $< 1$: revenire la medie')))

EF = ['btc', 'eth', 'sp500']
D.frame(T('Variance-Ratio Tests on the Whole Sample', 'Testele raportului varianțelor pe întregul eșantion'), table(
    'lrrrrrr', T('& $\\hat\\rho(1)$ & $Q(10)$ (p) & VR(2) & VR(5) & VR(10) & $Z^*(5)$ (p)', '& $\\hat\\rho(1)$ & $Q(10)$ (p) & VR(2) & VR(5) & VR(10) & $Z^*(5)$ (p)'),
    [f'{NM[k]} & $@{{e.{k}.rho1}}$ & $@{{e.{k}.lb}}$ (@{{e.{k}.lbp}}) & $@{{e.{k}.vr2}}$ & $@{{e.{k}.vr5}}$ & $@{{e.{k}.vr10}}$ & $@{{e.{k}.zs5}}$ (@{{e.{k}.p5}})' for k in EF],
    size='scriptsize') + table(
    'llrrrr', T('& period & $n$ & $\\hat\\rho(1)$ & VR(5) & $Z^*(5)$ (p)', '& perioada & $n$ & $\\hat\\rho(1)$ & VR(5) & $Z^*(5)$ (p)'),
    [f'{NM[k]} & @{{e.{k}.{y}.lab}} & @{{e.{k}.{y}.n}} & $@{{e.{k}.{y}.rho1}}$ & $@{{e.{k}.{y}.vr5}}$ & $@{{e.{k}.{y}.zs5}}$ (@{{e.{k}.{y}.p5}})'
     for k, y in [('btc', '2014'), ('btc', '2018'), ('btc', '2022'), ('eth', '2016'), ('eth', '2018'), ('eth', '2022')]],
    size='scriptsize') + items(
    T('Interpretation: no VR test rejects the random walk, in any sub-period; the Ljung--Box test rejects weakly for crypto because it ignores volatility clustering',
      'Interpretare: niciun test VR nu respinge mersul aleator, în nicio subperioadă; testul Ljung--Box respinge slab la cripto deoarece ignoră volatility clustering')), 'scriptsize')

chart(T('Efficiency Through Time', 'Eficiența în timp'), 'sfm_ch14_rolling_vr', 'SFM_ch14_garch_efficiency', [
    T('Robust $Z^*(5)$ on windows of 500 observations, one window every 21 observations (as in Chapter 7)', 'Statistica robustă $Z^*(5)$ pe ferestre de 500 de observații, o fereastră la fiecare 21 de observații (ca în Capitolul 7)'),
    T('Interpretation: Bitcoin @{rvr.btc.rej}\\% of @{rvr.btc.n} windows reject; with the i.i.d.\\ statistic $Z(5)$ the share would be @{rvr.btc.iid}\\% for Bitcoin and @{rvr.sp500.iid}\\% for the S\\&P 500: ignoring volatility clustering creates false rejections',
      'Interpretare: la Bitcoin, @{rvr.btc.rej}\\% din @{rvr.btc.n} de ferestre resping; cu statistica i.i.d.\\ $Z(5)$, proporția ar fi @{rvr.btc.iid}\\% pentru Bitcoin și @{rvr.sp500.iid}\\% pentru S\\&P 500: ignorarea volatility clustering produce respingeri false')],
    h='0.46\\textheight')

D.frame(T('How Efficiency Evolved', 'Evoluția eficienței'), items(
    (T('\\refUrquhart: Bitcoin was inefficient in 2010--2016 as a whole, but moved towards efficiency after 2013', '\\refUrquhart: Bitcoin a fost ineficient în 2010--2016 ca întreg, dar s-a apropiat de eficiență după 2013'),
     [T('\\refNC: a power transformation of the same returns passes the tests: the early verdict was fragile', '\\refNC: o transformare de tip putere a acelorași randamente trece testele: verdictul inițial era fragil'),
      T('\\refTL: an adjusted measure of inefficiency; crypto markets became more efficient after 2017', '\\refTL: o măsură ajustată a ineficienței; piețele cripto au devenit mai eficiente după 2017')]),
    (T('Our data (since 2014): VR(5) between @{rvr.btc.vmin} and @{rvr.btc.vmax} on rolling windows; no robust rejection', 'Datele noastre (din 2014): VR(5) între @{rvr.btc.vmin} și @{rvr.btc.vmax} pe ferestre mobile; nicio respingere robustă'),
     [T('the market grew, arbitrageurs arrived, derivatives and ETFs appeared: the adaptive-markets view of Chapter 11', 'piața s-a extins, au intrat arbitrajorii, au apărut derivatele și ETF-urile: perspectiva piețelor adaptive din Capitolul 11')]),
    T('Weak-form efficiency is not efficiency in general: prices can be unpredictable and still be far from any fundamental value', 'Eficiența în formă slabă nu este eficiență în general: prețurile pot fi imprevizibile și totuși departe de orice valoare fundamentală')))

D.frame(T('Case Study: Trading and Arbitrage Across Exchanges', 'Studiu de caz: tranzacționare și arbitraj între exchange-uri'), items(
    (T('\\refMS: tick data of 34 exchanges in 19 countries, 2017--2018', '\\refMS: date tick-by-tick de pe 34 de exchange-uri din 19 țări, 2017--2018'),
     [T('\\textbf{arbitrage}: buy where the price is low, sell where it is high; in an efficient market the gap closes fast', '\\textbf{arbitraj}: cumpărăm unde prețul este mic, vindem unde este mare; pe o piață eficientă diferența se închide repede')]),
    (T('Findings', 'Rezultatele'),
     [T('large and persistent price gaps between countries (Korea, Japan and the United States), much smaller within a country', 'diferențe mari și persistente de preț între țări (Coreea, Japonia și SUA), mult mai mici în interiorul unei țări'),
      T('the gaps open when Bitcoin rises and capital controls make the arbitrage hard', 'diferențele se deschid cînd Bitcoin crește, iar controlul capitalurilor face arbitrajul dificil'),
      T('a common component explains most of the order flow: prices are driven by one global market', 'o componentă comună explică cea mai mare parte a fluxului de ordine: prețurile sînt determinate de o piață globală')]),
    T('Lesson: the ``law of one price\'\' fails where money cannot move freely; a statistical test on one series cannot see this', 'Lecția: „legea prețului unic” nu funcționează acolo unde banii nu circulă liber; un test statistic pe o singură serie nu poate vedea asta')))

D.frame(T('Case Study: Is Bitcoin Really Untethered?', 'Studiu de caz: este Bitcoin cu adevărat independent de Tether?'), items(
    (T('\\refGS: blockchain records of Bitcoin and of the stablecoin Tether (USDT), 2017--2018', '\\refGS: înregistrările din blockchain ale Bitcoin și ale stablecoin-ului Tether (USDT), 2017--2018'),
     [T('question: was new USDT used to buy Bitcoin after price falls?', 'întrebarea: a fost folosit USDT nou emis pentru a cumpăra Bitcoin după scăderi de preț?')]),
    (T('Findings', 'Rezultatele'),
     [T('purchases with Tether came after market falls and were followed by price rises', 'cumpărările cu Tether au venit după scăderi ale pieței și au fost urmate de creșteri ale prețului'),
      T('the flows came mostly from one large account on one exchange (Bitfinex)', 'fluxurile proveneau în principal de la un singur cont mare de pe un singur exchange (Bitfinex)'),
      T('hours with such flows account for a large share of the 2017 rise', 'orele cu astfel de fluxuri explică o parte mare a creșterii din 2017')]),
    T('Related: manipulation of the Mt.~Gox exchange in 2013 \\refGHMO. Lesson: blockchain data allow tests that are impossible in equity markets', 'Similar: manipularea exchange-ului Mt.~Gox în 2013 \\refGHMO. Lecția: datele din blockchain permit teste imposibile pe piețele de acțiuni')))

D.recap(('Is the Crypto Market Efficient?', 'este piața cripto eficientă?'), [
    T('Robust VR tests do not reject the random walk for Bitcoin and Ethereum since 2014', 'Testele VR robuste nu resping mersul aleator pentru Bitcoin și Ethereum din 2014'),
    T('Early studies found inefficiency; the market became more efficient as it grew', 'Studiile timpurii au găsit ineficiență; piața a devenit mai eficientă pe măsură ce a crescut'),
    T('Arbitrage gaps across countries and manipulation show the limits of weak-form tests', 'Diferențele de arbitraj între țări și manipularea arată limitele testelor de formă slabă')])

# =============================================================================
# 5. CORELAȚIA CU ACȚIUNILE ȘI AURUL
# =============================================================================
D.section('Correlation with Equities and Gold', 'Corelația cu acțiunile și aurul')

D.frame(T('Measuring Correlation Across Calendars', 'Măsurarea corelației între calendare diferite'), items(
    (T('Bitcoin trades 7 days a week, the S\\&P 500 and gold 5: first join the \\textbf{prices} on common days, then compute returns', 'Bitcoin se tranzacționează 7 zile pe săptămînă, S\\&P 500 și aurul 5: întîi unim \\textbf{prețurile} în zilele comune, apoi calculăm randamentele'),
     [T('Monday\'s Bitcoin return then covers Friday to Monday, like the S\\&P 500 return', 'randamentul Bitcoin de luni acoperă atunci perioada vineri--luni, ca și randamentul S\\&P 500'),
      T('joining returns instead would drop the weekend moves of Bitcoin', 'unirea randamentelor, în schimb, ar elimina mișcările Bitcoin din weekend')]),
    (T('\\textbf{Asynchronous} closes: Bitcoin at midnight UTC, the S\\&P 500 at 16:00 New York time', 'Închideri \\textbf{asincrone}: Bitcoin la miezul nopții UTC, S\\&P 500 la ora 16:00 la New York'),
     [T('weekly returns (Friday to Friday) reduce this problem; we report both', 'randamentele săptămînale (vineri--vineri) reduc această problemă; le raportăm pe ambele')]),
    (T('\\textbf{Rolling correlation}: the sample correlation on the last 250 common days, recomputed every day', '\\textbf{Corelația mobilă}: corelația de selecție pe ultimele 250 de zile comune, recalculată în fiecare zi'),
     [T('standard error about $(1 - \\rho^2)/\\sqrt{250} \\approx 0.06$ ($\\rho$: the true correlation, here small): moves below 0.1 are noise', 'eroarea standard este circa $(1 - \\rho^2)/\\sqrt{250} \\approx 0{,}06$ ($\\rho$: corelația adevărată, aici mică): mișcările sub 0,1 sînt zgomot')])))

chart(T('Rolling Correlation with Equities and Gold', 'Corelația mobilă cu acțiunile și aurul'), 'sfm_ch14_rolling_corr', 'SFM_ch14_correlation', [
    T('Daily log returns on common days; 250-day windows; dashed: COVID-19, Terra/Luna, FTX and the spot ETFs', 'Randamente logaritmice zilnice în zilele comune; ferestre de 250 de zile; linii întrerupte: COVID-19, Terra/Luna, FTX și ETF-urile spot'),
    T('Interpretation: Bitcoin and the S\\&P 500 were uncorrelated before 2020 (@{c.btc_sp500.pre2020}) and positively correlated after (@{c.btc_sp500.post2020}); the maximum, @{c.btc_sp500.max}, came on @{c.btc_sp500.maxd}; gold stays weakly linked',
      'Interpretare: Bitcoin și S\\&P 500 erau necorelate înainte de 2020 (@{c.btc_sp500.pre2020}) și corelate pozitiv după (@{c.btc_sp500.post2020}); maximul, @{c.btc_sp500.max}, a apărut pe @{c.btc_sp500.maxd}; legătura cu aurul rămîne slabă')],
    h='0.48\\textheight')

D.frame(T('Correlation Year by Year', 'Corelația an cu an'), table(
    'l' + 'r' * 7, T('Bitcoin with & 2019 & 2020 & 2021 & 2022 & 2023 & 2024 & 2025', 'Bitcoin cu & 2019 & 2020 & 2021 & 2022 & 2023 & 2024 & 2025'),
    ['S\\&P 500 & ' + ' & '.join(f'$@{{c.btc_sp500.y{y}}}$' for y in range(2019, 2026)),
     T('gold', 'aurul') + ' & ' + ' & '.join(f'$@{{c.btc_gold.y{y}}}$' for y in range(2019, 2026)),
     T('Ethereum with S\\&P 500', 'Ethereum cu S\\&P 500') + ' & ' + ' & '.join(f'$@{{c.eth_sp500.y{y}}}$' for y in range(2019, 2026))],
    size='scriptsize') + items(
    (T('Whole period: daily @{c.btc_sp500.all}, weekly @{c.btc_sp500.weekly} for Bitcoin and the S\\&P 500', 'Întreaga perioadă: zilnic @{c.btc_sp500.all}, săptămînal @{c.btc_sp500.weekly} pentru Bitcoin și S\\&P 500'),
     [T('the asynchronous closes do not explain the correlation: weekly returns give a similar number', 'închiderile asincrone nu explică corelația: randamentele săptămînale dau o valoare apropiată')]),
    (T('Interpretation', 'Interpretare'),
     [T('since 2020, crypto behaves as a risk asset: it falls with equities when interest rates rise (2022)', 'din 2020, activele cripto se comportă ca active riscante: scad odată cu acțiunile cînd dobînzile cresc (2022)'),
      T('the change coincides with the arrival of institutional investors; a diversification benefit measured before 2020 no longer holds', 'schimbarea coincide cu intrarea investitorilor instituționali; beneficiul diversificării măsurat înainte de 2020 nu se mai păstrează')])), 'footnotesize')

D.frame(T('Three Crisis Episodes', 'Trei episoade de criză'), table(
    'llrrrrr', T('Episode & period & Bitcoin & Ethereum & S\\&P 500 & gold & worst Bitcoin day', 'Episodul & perioada & Bitcoin & Ethereum & S\\&P 500 & aur & cea mai proastă zi Bitcoin'),
    [T('COVID-19 & @{cr.covid.a} -- @{cr.covid.b}', 'COVID-19 & @{cr.covid.a} -- @{cr.covid.b}') + ' & $@{cr.covid.btc}$ & $@{cr.covid.eth}$ & $@{cr.covid.sp500}$ & $@{cr.covid.gold}$ & $@{cr.covid.worst}$',
     T('Terra/Luna & @{cr.terra.a} -- @{cr.terra.b}', 'Terra/Luna & @{cr.terra.a} -- @{cr.terra.b}') + ' & $@{cr.terra.btc}$ & $@{cr.terra.eth}$ & $@{cr.terra.sp500}$ & $@{cr.terra.gold}$ & $@{cr.terra.worst}$',
     T('FTX & @{cr.ftx.a} -- @{cr.ftx.b}', 'FTX & @{cr.ftx.a} -- @{cr.ftx.b}') + ' & $@{cr.ftx.btc}$ & $@{cr.ftx.eth}$ & $@{cr.ftx.sp500}$ & $@{cr.ftx.gold}$ & $@{cr.ftx.worst}$'],
    size='scriptsize') + items(
    T('Cumulative log returns in \\% from the first to the last close; worst day: the smallest daily log return of Bitcoin', 'Randamente logaritmice cumulate în \\% de la prima la ultima închidere; cea mai proastă zi: cel mai mic randament logaritmic zilnic al Bitcoin'),
    (T('Interpretation', 'Interpretare'),
     [T('COVID-19 (a global shock): crypto fell as much as equities, gold did not move', 'COVID-19 (un șoc global): activele cripto au scăzut cît acțiunile, aurul nu s-a mișcat'),
      T('Terra/Luna (May 2022) and FTX (an exchange that went bankrupt in November 2022): crypto-specific shocks; crypto fell by a quarter or more, the S\\&P 500 much less or not at all', 'Terra/Luna (mai 2022) și FTX (un exchange care a dat faliment în noiembrie 2022): șocuri specifice cripto; activele cripto au scăzut cu un sfert sau mai mult, S\\&P 500 mult mai puțin sau deloc'),
      T('the dependence is asymmetric: equity crashes reach crypto, crypto crashes rarely reach equities', 'dependența este asimetrică: crahurile acțiunilor ajung la cripto, crahurile cripto rareori ajung la acțiuni')])), 'footnotesize')

D.frame(T('Hedge or Safe Haven?', 'Acoperire sau refugiu?'), items(
    (T('\\refBL: a \\textbf{hedge} is uncorrelated or negatively correlated with equities on average; a \\textbf{safe haven} is so in market crashes', '\\refBL: o \\textbf{acoperire} (hedge) este necorelată sau corelată negativ cu acțiunile în medie; un \\textbf{refugiu} (safe haven) este astfel în timpul crahurilor'),
     [T('test: the mean return on the worst 5\\% of S\\&P 500 days (@{w.btc.nb} common days, S\\&P 500 below $@{w.btc.thr}\\%$, on average $@{w.btc.msp}\\%$)', 'testul: randamentul mediu în cele mai proaste 5\\% dintre zilele S\\&P 500 (@{w.btc.nb} de zile comune, S\\&P 500 sub $@{w.btc.thr}\\%$, în medie $@{w.btc.msp}\\%$)')]),
    (T('Bitcoin: $@{w.btc.m}\\%$ on those days ($t = @{w.btc.t}$, p @{w.btc.p}); negative on @{w.btc.neg}\\% of them', 'Bitcoin: $@{w.btc.m}\\%$ în acele zile ($t = @{w.btc.t}$, p @{w.btc.p}); negativ în @{w.btc.neg}\\% dintre ele'),
     [T('gold: @{w.gold.m}\\% ($t = @{w.gold.t}$, p @{w.gold.p})', 'aurul: @{w.gold.m}\\% ($t = @{w.gold.t}$, p @{w.gold.p})')]),
    T('Interpretation: Bitcoin is not a safe haven; on the worst equity days it falls as much as the S\\&P 500; gold holds its value', 'Interpretare: Bitcoin nu este un activ de refugiu; în cele mai proaste zile ale acțiunilor scade cît S\\&P 500; aurul își păstrează valoarea'),
    T('The ``digital gold\'\' narrative is not supported by the data of this sample', 'Narațiunea „aurului digital” nu este susținută de datele acestui eșantion')))

D.recap(('Correlation with Equities and Gold', 'corelația cu acțiunile și aurul'), [
    T('Join prices on common days, then compute returns; check weekly returns', 'Unim prețurile în zilele comune, apoi calculăm randamentele; verificăm și randamentele săptămînale'),
    T('Correlation with equities: about 0 before 2020, about @{c.btc_sp500.post2020} after', 'Corelația cu acțiunile: circa 0 înainte de 2020, circa @{c.btc_sp500.post2020} după'),
    T('Bitcoin is neither a hedge nor a safe haven for equities in this sample', 'Bitcoin nu este nici acoperire, nici refugiu pentru acțiuni în acest eșantion')])

# =============================================================================
# 6. BULE ȘI CRAHURI
# =============================================================================
D.section('Bubbles and Crashes', 'Bule și crahuri')

D.frame(T('Drawdown', 'Drawdown-ul'), items(
    (T('\\textbf{Drawdown}: $D_t = P_t/\\max_{s \\le t} P_s - 1$, the loss from the last record high (Chapter 1)', '\\textbf{Drawdown}: $D_t = P_t/\\max_{s \\le t} P_s - 1$, pierderea față de ultimul maxim istoric (Capitolul 1)'),
     [T('$P_t$: the price on day $t$; $\\max_{s \\le t} P_s$: the highest price up to day $t$; $D_t \\le 0$', '$P_t$: prețul din ziua $t$; $\\max_{s \\le t} P_s$: cel mai mare preț pînă în ziua $t$; $D_t \\le 0$'),
      T('\\textbf{maximum drawdown}: $\\min_t D_t$; \\textbf{time under water}: the days until a new record high', '\\textbf{drawdown maxim}: $\\min_t D_t$; \\textbf{perioada underwater} (time under water): zilele pînă la un nou maxim istoric')]),
    (T('A loss of $d$ needs a gain of $d/(1 - d)$ to recover, because $(1 - d)(1 + g) = 1$ gives $g = d/(1 - d)$', 'O pierdere $d$ cere un cîștig $d/(1 - d)$ pentru revenirea la vîrful anterior, deoarece $(1 - d)(1 + g) = 1$ dă $g = d/(1 - d)$'),
     [T('$d = 75\\%$: $0.75/0.25 = 300\\%$; $d = 80\\%$: @{dd.ex80}\\%', '$d = 75\\%$: $0{,}75/0{,}25 = 300\\%$; $d = 80\\%$: @{dd.ex80}\\%'),
      T('Bitcoin\'s deepest drawdown in our data: $@{dd.btc.mdd}\\%$ (@{dd.btc.mddd}); recovery needed a gain of @{dd.big.need}\\%', 'cel mai adînc drawdown al Bitcoin în datele noastre: $@{dd.btc.mdd}\\%$ (@{dd.btc.mddd}); recuperarea a cerut un cîștig de @{dd.big.need}\\%')]),
    T('Drawdowns depend on the path, not only on the distribution of returns: high volatility and clustering produce deep ones', 'Drawdown-urile depind de traiectorie, nu doar de distribuția randamentelor: volatilitatea mare și volatility clustering produc drawdown-uri adînci')))

chart(T('Drawdowns of Bitcoin, Ethereum and the S\\&P 500', 'Drawdown-urile Bitcoin, Ethereum și S\\&P 500'), 'sfm_ch14_drawdowns', 'SFM_ch14_bubbles_etf', [
    T('Daily closes since 2016; maximum drawdowns: Bitcoin $@{dd.btc.mdd}\\%$, Ethereum $@{dd.eth.mdd}\\%$, S\\&P 500 $@{dd.sp500.mdd}\\%$', 'Închideri zilnice din 2016; drawdown-uri maxime: Bitcoin $@{dd.btc.mdd}\\%$, Ethereum $@{dd.eth.mdd}\\%$, S\\&P 500 $@{dd.sp500.mdd}\\%$'),
    T('Interpretation: Bitcoin spent @{dd.btc.s10}\\% of the days more than 10\\% below its record, Ethereum @{dd.eth.s10}\\%, the S\\&P 500 @{dd.sp500.s10}\\%; on @{end} Bitcoin was @{dd.btc.lastabs}\\% below its record',
      'Interpretare: Bitcoin a petrecut @{dd.btc.s10}\\% din zile la peste 10\\% sub maximul istoric, Ethereum @{dd.eth.s10}\\%, S\\&P 500 @{dd.sp500.s10}\\%; pe @{end}, Bitcoin era cu @{dd.btc.lastabs}\\% sub maximul istoric')],
    h='0.46\\textheight')

eprows = []
for e in ep:
    rec = e['recovery']
    eprows.append(f"⟦{e['peak']}||{e['peak']}⟧ & {money(e['peak_price'])} & {e['trough']} & {money(e['trough_price'])} & $⁅{e['depth']:.1f}⁆$ & "
                  + (rec if rec else T('not yet', 'încă nu')) + ' & ' + (str(e['days_under']) if e['days_under'] else '--'))
D.frame(T('Bitcoin\'s Large Drawdowns', 'Marile drawdown-uri ale Bitcoin'), table(
    'lrlrrlr', T('Peak & USD & trough & USD & depth (\\%) & new record & days under water', 'Maxim & USD & minim & USD & adîncime (\\%) & nou maxim & zile underwater'),
    eprows, size='scriptsize') + items(
    T('Episodes since September 2014 whose drawdown exceeded 50\\%; dates as year-month-day', 'Episoadele din septembrie 2014 cu drawdown de peste 50\\%; datele în formatul an-lună-zi'),
    (T('Interpretation', 'Interpretare'),
     [T('@{dd.nep} falls of more than half in twelve years: a crash of this size is a regular event, not a rare one', '@{dd.nep} scăderi de peste jumătate în doisprezece ani: un crah de această mărime este un eveniment obișnuit, nu unul rar'),
      T('the 2017 and 2021 peaks were recovered after @{dd.d17} and @{dd.d21} days', 'maximele din 2017 și 2021 au fost recuperate după @{dd.d17} și, respectiv, @{dd.d21} de zile')])), 'footnotesize')

D.frame(T('Bubbles and the Log-Periodic Idea', 'Bulele și ideea log-periodică'), cols(items(
    (T('\\textbf{Bubble}: a price far above the fundamental value, sustained by the expectation of selling higher', '\\textbf{Bulă}: un preț mult peste valoarea fundamentală, susținut de așteptarea unei revînzări la un preț mai mare'),
     [T('\\refCF: the fundamental value of Bitcoin is zero; its price shows speculative bubbles', '\\refCF: valoarea fundamentală a Bitcoin este zero; prețul său arată bule speculative')]),
    (T('Log-periodic power law (LPPL) \\refJLS: before a crash, $\\ln P_t \\approx A + B(t_c - t)^m + C(t_c - t)^m\\cos(\\omega\\ln(t_c - t) - \\phi)$', 'Legea de putere log-periodică (LPPL) \\refJLS: înaintea unui crah, $\\ln P_t \\approx A + B(t_c - t)^m + C(t_c - t)^m\\cos(\\omega\\ln(t_c - t) - \\phi)$'),
     [T('$t_c$: the critical time (the most likely crash date); $A$: the log price reached at $t_c$ (here $A$ is not the number of periods per year)', '$t_c$: momentul critic (data cea mai probabilă a crahului); $A$: logaritmul prețului atins la $t_c$ (aici $A$ nu este numărul de perioade pe an)'),
      T('$B < 0$, $0 < m < 1$: faster than exponential growth; $C$, $\\omega$, $\\phi$: amplitude, frequency and phase of oscillations that accelerate towards $t_c$', '$B < 0$, $0 < m < 1$: o creștere mai rapidă decît cea exponențială; $C$, $\\omega$, $\\phi$: amplitudinea, frecvența și faza unor oscilații tot mai dese spre $t_c$')]),
    T('Caution: seven parameters, many local optima, and $t_c$ is the most uncertain of them; fits look convincing only after the crash', 'Atenție: șapte parametri, multe optime locale, iar $t_c$ este cel mai incert dintre ei; ajustările par convingătoare abia după crah')),
    ph('sornette', T('Didier Sornette', 'Didier Sornette'), h='0.30\\textheight'), wl='0.66', wr='0.31'), 'footnotesize')

D.recap(('Bubbles and Crashes', 'bule și crahuri'), [
    T('Drawdown $= P_t/\\max P - 1$; a fall of $d$ needs a gain of $d/(1 - d)$', 'Drawdown $= P_t/\\max P - 1$; o scădere $d$ cere un cîștig $d/(1 - d)$'),
    T('Bitcoin lost more than half of its value @{dd.nep} times since 2014', 'Bitcoin a pierdut mai mult de jumătate din valoare de @{dd.nep} ori din 2014'),
    T('Bubble models are useful descriptions, unreliable for timing a crash', 'Modelele de bule sînt descrieri utile, dar nesigure pentru datarea unui crah')])

# =============================================================================
# 7. ETF-URILE SPOT PE BITCOIN
# =============================================================================
D.section('Spot Bitcoin ETFs', 'ETF-urile spot pe Bitcoin')

D.frame(T('January 2024: Bitcoin Enters the Stock Exchange', 'Ianuarie 2024: Bitcoin intră la bursă'), cols(items(
    (T('\\textbf{ETF} (exchange-traded fund): a fund whose shares trade on a stock exchange; \\textbf{spot}: it holds the asset itself, not futures', '\\textbf{ETF} (exchange-traded fund): un fond ale cărui unități se tranzacționează la bursă; \\textbf{spot}: deține activul însuși, nu contracte futures'),
     [T('10 January 2024: the SEC (Securities and Exchange Commission) approves 11 spot Bitcoin ETFs \\refSEC; trading starts on 11 January', '10 ianuarie 2024: SEC (Securities and Exchange Commission) aprobă 11 ETF-uri spot pe Bitcoin \\refSEC; tranzacționarea începe pe 11 ianuarie')]),
    (T('Why it matters', 'Importanța momentului'),
     [T('pension funds and advisers can hold Bitcoin through a regulated security', 'fondurile de pensii și consultanții pot deține Bitcoin printr-un instrument financiar reglementat'),
      T('ETF shares trade only on weekdays: part of the trading moves to stock-market hours', 'unitățile ETF se tranzacționează doar în zilele lucrătoare: o parte a tranzacționării se mută în orele bursei')]),
    T('Statistical question: did Bitcoin\'s volatility, its weekend pattern or its link with equities change?', 'Întrebarea statistică: s-au schimbat volatilitatea Bitcoin, tiparul ei de weekend sau legătura cu acțiunile?')),
    ph('sec', T('SEC headquarters, Washington, D.C.', 'Sediul SEC, Washington, D.C.'), h='0.34\\textheight'), wl='0.62', wr='0.35'), 'footnotesize')

chart(T('Volatility and the Weekend Share Around the ETFs', 'Volatilitatea și ponderea weekendului în jurul ETF-urilor'), 'sfm_ch14_etf', 'SFM_ch14_bubbles_etf', [
    T('Left: Bitcoin volatility on 90-day windows, annualised with $\\sqrt{365}$. Right: weekend share of the sum of squared returns by year (2/7 if every day had the same variance)',
      'Stînga: volatilitatea Bitcoin pe ferestre de 90 de zile, anualizată cu $\\sqrt{365}$. Dreapta: ponderea weekendului în suma pătratelor randamentelor, pe ani (2/7 dacă fiecare zi ar avea aceeași varianță)'),
    T('Interpretation: volatility kept falling after January 2024, but it was already falling before; the weekend share has been well below 2/7 since 2020', 'Interpretare: volatilitatea a continuat să scadă după ianuarie 2024, dar era deja în scădere înainte; ponderea weekendului este mult sub 2/7 din 2020')],
    h='0.46\\textheight')

D.frame(T('Before and After: a Simple Comparison', 'Înainte și după: o comparație simplă'), table(
    'lrr', T('Bitcoin & two years before & two years after', 'Bitcoin & doi ani înainte & doi ani după'),
    [T('annualised volatility (\\%)', 'volatilitatea anuală (\\%)') + ' & $@{etf.vol_before}$ & $@{etf.vol_after}$',
     T('weekend/weekday variance', 'varianța weekend/zile lucrătoare') + ' & $@{etf.we_ratio_before}$ & $@{etf.we_ratio_after}$',
     T('correlation with the S\\&P 500', 'corelația cu S\\&P 500') + ' & $@{etf.corr_before}$ & $@{etf.corr_after}$',
     T('GARCH(1,1)-t $\\hat\\alpha + \\hat\\beta$', 'GARCH(1,1)-t $\\hat\\alpha + \\hat\\beta$') + ' & $@{etf.pers_before}$ & $@{etf.pers_after}$'],
    size='footnotesize') + items(
    T('Windows: 11 January 2022 -- 10 January 2024 (@{etf.nb} days) and 11 January 2024 -- 10 January 2026 (@{etf.na} days)', 'Ferestrele: 11 ianuarie 2022 -- 10 ianuarie 2024 (@{etf.nb} de zile) și 11 ianuarie 2024 -- 10 ianuarie 2026 (@{etf.na} de zile)'),
    T('Brown--Forsythe test of equal variance: statistic @{etf.bf}, p $= @{etf.p_bf}$: the fall in volatility is not significant', 'Testul Brown--Forsythe al egalității varianțelor: statistica @{etf.bf}, p $= @{etf.p_bf}$: scăderea volatilității nu este semnificativă'),
    (T('Interpretation', 'Interpretare'),
     [T('a before--after comparison is not a causal effect: interest rates, the halving of April 2024 and the US election changed at the same time', 'o comparație înainte--după nu este un efect cauzal: dobînzile, halving-ul din aprilie 2024 și alegerile din SUA s-au schimbat în același timp'),
      T('a credible study needs a control group (e.g.\\ coins without an ETF) or intraday data around the opening of the stock exchange', 'un studiu credibil are nevoie de un grup de control (de exemplu monede fără ETF) sau de date intrazilnice în jurul deschiderii bursei')])), 'footnotesize')

D.frame(T('How Well Does an ETF Track Bitcoin?', 'Cît de bine urmărește un ETF prețul Bitcoin?'), items(
    (T('IBIT (iShares Bitcoin Trust), adjusted close, and Bitcoin, prices joined on the @{ibit.n} trading days of the ETF', 'IBIT (iShares Bitcoin Trust), închiderea ajustată, și Bitcoin, prețuri unite în cele @{ibit.n} zile de tranzacționare ale ETF-ului'),
     [T('correlation of daily returns @{ibit.corr}; slope of IBIT on Bitcoin @{ibit.beta}', 'corelația randamentelor zilnice @{ibit.corr}; panta IBIT în funcție de Bitcoin @{ibit.beta}'),
      T('\\textbf{tracking error}: the annualised standard deviation of the return difference, @{ibit.te}\\%', '\\textbf{tracking error}: abaterea standard anualizată a diferenței de randament, @{ibit.te}\\%')]),
    (T('Why not 1: the two closes are four hours apart (16:00 in New York against midnight UTC)', 'Explicația abaterii de la 1: cele două închideri sînt la patru ore distanță (ora 16:00 la New York față de miezul nopții UTC)'),
     [T('over a week the difference averages out; the fund itself holds Bitcoin one to one, minus a small fee', 'pe o săptămînă diferența se compensează; fondul deține Bitcoin unu la unu, minus un comision mic')]),
    T('Lesson: a high tracking error can be a measurement artefact; align the times before judging a fund', 'Lecția: un tracking error mare poate fi un artefact de măsurare; aliniem orele înainte de a judeca un fond')))

D.recap(('Spot Bitcoin ETFs', 'ETF-urile spot pe Bitcoin'), [
    T('January 2024: Bitcoin becomes available through regulated ETFs', 'Ianuarie 2024: Bitcoin devine accesibil prin ETF-uri reglementate'),
    T('Volatility fell (@{etf.vol_before}\\% to @{etf.vol_after}\\%), not significantly; causality is not identified', 'Volatilitatea a scăzut (de la @{etf.vol_before}\\% la @{etf.vol_after}\\%), nesemnificativ; cauzalitatea nu este identificată'),
    T('Asynchronous closing times inflate the tracking error of daily returns', 'Orele de închidere asincrone măresc tracking error-ul randamentelor zilnice')])

# =============================================================================
# 8. STABLECOINS
# =============================================================================
D.section('Stablecoins', 'Stablecoins')

D.frame(T('What Is a Stablecoin?', 'Stablecoins: definiție și tipuri'), table(
    TB + 'p{2.2cm}' + TB + 'p{4.4cm}' + TB + 'p{2.0cm}' + TB + 'p{2.4cm}',
    T('Type & how the peg is held & examples & risk', 'Tipul & cum se menține paritatea & exemple & riscul'),
    [T('fiat-backed & reserves of dollars, deposits and Treasury bills; holders redeem 1 coin for 1 USD at the issuer & USDT, USDC & the reserves and the bank that holds them',
       'garantat cu monedă fiduciară (fiat-backed) & rezerve în dolari, depozite și titluri de stat; deținătorii răscumpără 1 monedă pentru 1 USD la emitent & USDT, USDC & rezervele și banca la care sînt păstrate'),
     T('crypto-backed & over-collateralised loans in crypto, liquidated automatically by smart contracts & DAI & a crash of the collateral',
       'garantat cu active cripto (crypto-backed) & credite supra-garantate cu active cripto, lichidate automat de contracte inteligente & DAI & o prăbușire a garanției'),
     T('algorithmic & no reserves: an algorithm mints and burns a second coin to keep the price at 1 USD & UST (Terra) & a run that destroys both coins',
       'algoritmic & fără rezerve: un algoritm emite și retrage din circulație (burn) o a doua monedă pentru a ține prețul la 1 USD & UST (Terra) & o retragere masivă (run) care distruge ambele monede')],
    size='scriptsize') + items(
    (T('\\textbf{Stablecoin}: a crypto asset designed to keep a fixed price, usually 1 USD (the \\textbf{peg})', '\\textbf{Stablecoin}: un activ cripto conceput să păstreze un preț fix, de obicei 1 USD (\\textbf{paritatea}, peg)'),
     [T('used as cash on crypto exchanges and for payments across borders; DefiLlama classifies today @{sup.fiat}\\% of the supply as fiat-backed, @{sup.crypto}\\% as crypto-backed', 'folosit ca numerar pe exchange-uri și pentru plăți transfrontaliere; DefiLlama clasifică azi @{sup.fiat}\\% din ofertă ca garantată fiat și @{sup.crypto}\\% ca garantată cripto')])), 'footnotesize')

D.frame(T('How a Peg Is Held: Arbitrage', 'Menținerea parității: arbitrajul'), items(
    (T('Primary market: authorised traders create a coin for 1 USD at the issuer, or redeem it for 1 USD', 'Piața primară: tranzacționari autorizați creează o monedă pentru 1 USD la emitent sau o răscumpără pentru 1 USD'),
     [T('price 0.99 on exchanges: buy at 0.99, redeem at 1.00, profit 1 cent: demand pushes the price back up', 'prețul 0,99 pe exchange-uri: cumpărăm la 0,99, răscumpărăm la 1,00, profit 1 cent: cererea împinge prețul înapoi în sus'),
      T('price 1.01: create at 1.00, sell at 1.01: supply pushes the price down', 'prețul 1,01: creăm la 1,00, vindem la 1,01: oferta împinge prețul în jos')]),
    (T('\\refLVN: the peg of Tether holds because of this arbitrage; premiums appear when traders flee into stablecoins in crypto downturns', '\\refLVN: paritatea Tether se menține prin acest arbitraj; prime apar cînd investitorii se refugiază în stablecoins în perioadele de scădere cripto'),
     [T('\\refMZZ: with few authorised arbitrageurs, runs are less likely but deviations last longer', '\\refMZZ: cu puțini arbitrajori autorizați, retragerile masive sînt mai puțin probabile, dar abaterile durează mai mult')]),
    T('The arbitrage works only while traders trust that redemption at 1 USD will be honoured: a peg is a promise', 'Arbitrajul funcționează doar cîtă vreme investitorii cred că răscumpărarea la 1 USD va fi onorată: o paritate este o promisiune')))

chart(T('Supply of Stablecoins', 'Oferta de stablecoins'), 'sfm_ch14_stablecoin_supply', 'SFM_ch14_stablecoins', [
    T('Circulating supply in billion USD (DefiLlama): all USD stablecoins, USDT, USDC and DAI', 'Oferta în circulație, în miliarde USD (DefiLlama): toate stablecoin-urile în USD, USDT, USDC și DAI'),
    T('Interpretation: from @{sup.total_2020} billion USD at the start of 2020 to @{sup.total_end} billion on @{sup.end}, a factor of @{sup.growth_x}; USDT holds @{sup.usdt_share}\\%, USDC @{sup.usdc_share}\\%',
      'Interpretare: de la @{sup.total_2020} miliarde USD la începutul lui 2020 la @{sup.total_end} de miliarde pe @{sup.end}, de @{sup.growth_x} de ori mai mult; USDT deține @{sup.usdt_share}\\%, USDC @{sup.usdc_share}\\%'),
    T('The total fell from @{sup.total_2022max} billion (@{sup.maxd}) to @{sup.total_trough} billion (@{sup.trd}) after Terra and FTX; USDC fell from @{sup.usdc_2023_03_10} to @{sup.usdc_2023_12_31} billion in 2023',
      'Totalul a scăzut de la @{sup.total_2022max} de miliarde (@{sup.maxd}) la @{sup.total_trough} de miliarde (@{sup.trd}) după Terra și FTX; USDC a scăzut de la @{sup.usdc_2023_03_10} la @{sup.usdc_2023_12_31} de miliarde în 2023')],
    h='0.44\\textheight')

V.put('peg.ex.bp', 1e4 * (0.9715 - 1), 0)
D.frame(T('Peg Deviation in Basis Points', 'Abaterea de la paritate în puncte de bază'), items(
    (T('\\textbf{Basis point} (bp): one hundredth of a percent, 0.01\\% $= 0.0001$', '\\textbf{Punct de bază} (bp, basis point): o sutime de procent, 0,01\\% $= 0{,}0001$'),
     [T('\\textbf{peg deviation}: $d_t = 10\\,000\\,(P_t - 1)$ bp for a coin pegged to 1 USD; $P_t$: its price in USD on day $t$', '\\textbf{abaterea de la paritate}: $d_t = 10\\,000\\,(P_t - 1)$ bp pentru o monedă cu paritatea 1 USD; $P_t$: prețul ei în USD în ziua $t$'),
      T('$d_t < 0$: the coin trades below the peg; $d_t > 0$: above it', '$d_t < 0$: moneda se tranzacționează sub paritate; $d_t > 0$: peste paritate')]),
    (T('\\textbf{Worked example}: USDC closed at 0.9715 USD on 11 March 2023', '\\textbf{Exemplu lucrat}: USDC s-a închis la 0,9715 USD pe 11 martie 2023'),
     [T('$d = 10\\,000 \\times (0.9715 - 1) = @{peg.ex.bp}$ bp, a loss of 2.85\\%', '$d = 10\\,000 \\times (0{,}9715 - 1) = @{peg.ex.bp}$ bp, o pierdere de 2,85\\%'),
      T('a holder of 1\\,000\\,000 USDC who sold at that close lost @{svb.loss} USD; at the daily low (@{svb.usdc.low}): @{svb.losslow} USD', 'un deținător de 1\\,000\\,000 USDC care a vîndut la acea închidere a pierdut @{svb.loss} USD; la minimul zilei (@{svb.usdc.low}): @{svb.losslow} USD')]),
    (T('Statistics of a peg: the median absolute deviation, the share of days beyond 10 or 50 bp, the worst close and the worst low', 'Statisticile unei parități: abaterea absolută mediană, proporția zilelor peste 10 sau 50 bp, cea mai proastă închidere și cel mai prost minim'),
     [T('speed of return: AR(1) $d_t = \\phi\\,d_{t-1} + e_t$, half-life $\\ln 0.5/\\ln\\phi$ days', 'viteza revenirii: AR(1) $d_t = \\phi\\,d_{t-1} + e_t$, timpul de înjumătățire $\\ln 0{,}5/\\ln\\phi$ zile'),
      T('$\\phi \\in (0, 1)$: the share of today\'s deviation still present tomorrow; $e_t$: a new shock; a small $\\phi$ means a fast return to the peg', '$\\phi \\in (0, 1)$: fracțiunea din abaterea de azi care persistă mîine; $e_t$: un șoc nou; un $\\phi$ mic înseamnă o revenire rapidă la paritate')])))

chart(T('Peg Deviations of USDT, USDC and DAI', 'Abaterile de la paritate pentru USDT, USDC și DAI'), 'sfm_ch14_peg', 'SFM_ch14_stablecoins', [
    T('Daily closing deviation from 1 USD in bp since 2021 (EODHD)', 'Abaterea zilnică a închiderii de la 1 USD, în bp, din 2021 (EODHD)'),
    T('Median absolute deviation: USDT @{peg.usdt.mad} bp, USDC @{peg.usdc.mad} bp, DAI @{peg.dai.mad} bp; days beyond 10 bp: @{peg.usdt.gt10}\\%, @{peg.usdc.gt10}\\%, @{peg.dai.gt10}\\%',
      'Abaterea absolută mediană: USDT @{peg.usdt.mad} bp, USDC @{peg.usdc.mad} bp, DAI @{peg.dai.mad} bp; zile peste 10 bp: @{peg.usdt.gt10}\\%; @{peg.usdc.gt10}\\%; @{peg.dai.gt10}\\%'),
    T('Interpretation: half-lives of @{peg.usdt.hl}, @{peg.usdc.hl} and @{peg.dai.hl} days: deviations close within days; the exception is March 2023 (USDC $@{peg.usdc.min}$ bp, DAI $@{peg.dai.min}$ bp)',
      'Interpretare: timpi de înjumătățire de @{peg.usdt.hl}, @{peg.usdc.hl} și @{peg.dai.hl} zile: abaterile se închid în cîteva zile; excepția este martie 2023 (USDC $@{peg.usdc.min}$ bp, DAI $@{peg.dai.min}$ bp)')],
    h='0.48\\textheight')

D.frame(T('March 2023: USDC and Silicon Valley Bank', 'Martie 2023: USDC și Silicon Valley Bank'), cols(items(
    (T('10 March 2023: Silicon Valley Bank (SVB) is closed by its regulator \\refFDIC; part of the cash reserves of USDC was deposited there', '10 martie 2023: Silicon Valley Bank (SVB) este închisă de autoritatea de supraveghere \\refFDIC; o parte din rezervele în numerar ale USDC erau depuse acolo'),
     [T('a weekend follows: banks are closed, crypto markets are open', 'urmează un weekend: băncile sînt închise, piețele cripto sînt deschise')]),
    (T('11 March: USDC falls to @{svb.usdc.low} USD (low, $@{svb.usdc.lowbp}$ bp) and closes at @{svb.usdc.cmin}; DAI, backed largely by USDC, follows', '11 martie: USDC scade la @{svb.usdc.low} USD (minim, $@{svb.usdc.lowbp}$ bp) și se închide la @{svb.usdc.cmin}; DAI, garantat în mare parte cu USDC, îl urmează'),
     [T('USDT trades above 1 (close up to @{svb.usdt.cmaxbp} bp): money flees from one stablecoin into another', 'USDT se tranzacționează peste 1 (închidere de pînă la @{svb.usdt.cmaxbp} bp): capitalul se mută de la un stablecoin la altul')]),
    T('12 March, evening: US authorities guarantee all SVB deposits \\refFed; on 13 March USDC is back at the peg', '12 martie, seara: autoritățile americane garantează toate depozitele SVB \\refFed; pe 13 martie USDC revine la paritate')),
    ph('svb', T('SVB headquarters, Santa Clara, 13 March 2023', 'Sediul SVB, Santa Clara, 13 martie 2023'), h='0.34\\textheight'), wl='0.62', wr='0.35'), 'footnotesize')

chart(T('The USDC Depeg, Day by Day', 'Depeg-ul USDC, zi de zi'), 'sfm_ch14_svb', 'SFM_ch14_stablecoins', [
    T('Daily closes (circles) and lows (triangles), 8--20 March 2023; dashed: the peg', 'Închideri zilnice (cercuri) și minime (triunghiuri), 8--20 martie 2023; linia întreruptă: paritatea'),
    T('Interpretation: a fiat-backed coin is as safe as its bank: the peg broke when the reserves were in doubt and returned when the deposits were guaranteed; the run was on the reserves, not on the code',
      'Interpretare: o monedă garantată fiat este la fel de sigură ca banca ei: paritatea s-a rupt cînd rezervele au fost puse sub semnul întrebării și a revenit cînd depozitele au fost garantate; retragerea a vizat rezervele, nu codul')],
    h='0.48\\textheight')

D.frame(T('TerraUSD: an Algorithmic Stablecoin', 'TerraUSD: un stablecoin algoritmic'), items(
    (T('UST was kept at 1 USD by a swap with a second coin, LUNA: 1 UST could always be exchanged for 1 USD worth of LUNA', 'UST era ținut la 1 USD printr-un schimb cu o a doua monedă, LUNA: 1 UST putea fi schimbat oricînd pe LUNA în valoare de 1 USD'),
     [T('UST below 1: buy UST, swap it for 1 USD of new LUNA, sell the LUNA: UST is burned, LUNA is minted', 'UST sub 1: cumpărăm UST, îl schimbăm pe LUNA nou în valoare de 1 USD, vindem LUNA: UST este retras din circulație (burn), LUNA este emis'),
      T('the only backing was the market value of LUNA', 'singura garanție era valoarea de piață a LUNA')]),
    (T('Demand came from the Anchor protocol, which paid close to 20\\% a year on UST deposits \\refLMS', 'Cererea venea de la protocolul Anchor, care plătea aproape 20\\% pe an pentru depozitele în UST \\refLMS'),
     [T('UST supply peaked at @{ust.peak} billion on @{ust.peakd}', 'oferta de UST a atins maximul de @{ust.peak} miliarde pe @{ust.peakd}')]),
    (T('The \\textbf{death spiral}: large withdrawals from Anchor, UST below 1, more LUNA minted, LUNA falls, the backing shrinks, more withdrawals', '\\textbf{Spirala morții} (death spiral): retrageri mari din Anchor, UST sub 1, se emite mai mult LUNA, LUNA scade, garanția se micșorează, urmează și mai multe retrageri'),
     [T('\\refLMS: large, sophisticated holders withdrew first; small holders were the last to leave', '\\refLMS: deținătorii mari și sofisticați s-au retras primii; deținătorii mici au plecat ultimii')])))

chart(T('The Collapse of UST, May 2022', 'Prăbușirea UST, mai 2022'), 'sfm_ch14_ust', 'SFM_ch14_stablecoins', [
    T('DefiLlama: UST supply until the last published day (@{ust.lastd}) and the daily UST price', 'DefiLlama: oferta de UST pînă în ultima zi publicată (@{ust.lastd}) și prețul zilnic al UST'),
    T('Interpretation: the price was @{ust.px_0509} USD on 9 May, @{ust.px_0513} on 13 May, @{ust.px_0514} on 14 May and @{ust.px_0531} on 31 May; the supply fell by @{ust.drop}\\% in four days, from 7 to 11 May',
      'Interpretare: prețul era @{ust.px_0509} USD pe 9 mai, @{ust.px_0513} pe 13 mai, @{ust.px_0514} pe 14 mai și @{ust.px_0531} pe 31 mai; oferta a scăzut cu @{ust.drop}\\% în patru zile, între 7 și 11 mai'),
    T('Unlike USDC, there was no reserve and no lender of last resort: an algorithmic peg cannot be rescued', 'Spre deosebire de USDC, nu exista nicio rezervă și niciun creditor de ultimă instanță: o paritate algoritmică nu poate fi salvată')],
    h='0.44\\textheight')

D.frame(T('Stablecoins as Private Money', 'Stablecoins ca monedă privată'), items(
    (T('\\refGZ: stablecoins resemble the banknotes of private US banks in the ``wildcat banking\'\' era (1837--1863)', '\\refGZ: stablecoins seamănă cu bancnotele băncilor private americane din perioada „wildcat banking” (1837--1863)'),
     [T('notes of different banks traded at different discounts, depending on trust in each bank', 'bancnotele diferitelor bănci se tranzacționau cu reduceri diferite, după încrederea în fiecare bancă'),
      T('remedy then: a uniform national currency; remedy proposed now: regulate stablecoins as bank deposits', 'remediul de atunci: o monedă națională uniformă; remediul propus acum: reglementarea stablecoins ca depozite bancare')]),
    (T('\\refGorton: the peg holds as long as holders value the coin for trading with leverage in crypto markets', '\\refGorton: paritatea se menține cîtă vreme deținătorii prețuiesc moneda pentru tranzacții cu levier pe piețele cripto'),
     [T('when that demand disappears, the coin trades below par, like a bank note of an uncertain bank', 'cînd această cerere dispare, moneda se tranzacționează sub paritate, ca bancnota unei bănci nesigure')]),
    T('Statistical lesson: peg deviations are small and short in normal times and jump in a crisis: a heavy-tailed, regime-dependent series', 'Lecția statistică: abaterile de la paritate sînt mici și scurte în perioadele normale și cresc brusc în criză: o serie cu cozi groase, dependentă de regim')))

D.frame(T('Regulation in the European Union: MiCA', 'Reglementarea în Uniunea Europeană: MiCA'), cols(items(
    (T('MiCA, Markets in Crypto-Assets \\refMiCA, adopted in 2023', 'MiCA, Markets in Crypto-Assets \\refMiCA, adoptat în 2023'),
     [T('rules for stablecoins apply from 30 June 2024; for other crypto-asset services from 30 December 2024 \\refESMA', 'regulile pentru stablecoins se aplică din 30 iunie 2024; pentru celelalte servicii cu active cripto din 30 decembrie 2024 \\refESMA')]),
    (T('Stablecoins pegged to one currency are \\textbf{e-money tokens}', 'Stablecoins legate de o singură monedă sînt \\textbf{token-uri de monedă electronică} (e-money tokens)'),
     [T('issued only by authorised credit or e-money institutions; redemption at par at any time; no interest to holders', 'emise doar de instituții de credit sau de monedă electronică autorizate; răscumpărare la paritate oricînd; fără dobîndă pentru deținători'),
      T('reserves kept separate and invested in safe, liquid assets', 'rezervele sînt separate și investite în active sigure și lichide')]),
    T('Algorithmic stablecoins without reserves have no place in this framework', 'Stablecoins algoritmice fără rezerve nu au loc în acest cadru')),
    ph('ep', T('European Parliament, Strasbourg', 'Parlamentul European, Strasbourg'), h='0.30\\textheight'), wl='0.62', wr='0.35'), 'footnotesize')

D.frame(T('Regulation in the United States: the GENIUS Act', 'Reglementarea în SUA: GENIUS Act'), cols(items(
    (T('GENIUS Act (Guiding and Establishing National Innovation for U.S. Stablecoins Act), signed on 18 July 2025 \\refGENIUS', 'GENIUS Act (Guiding and Establishing National Innovation for U.S. Stablecoins Act), semnat pe 18 iulie 2025 \\refGENIUS'),
     [T('payment stablecoins: reserves of at least 1 USD per coin, in cash, deposits and short-term Treasury bills', 'stablecoins de plată: rezerve de cel puțin 1 USD pentru fiecare monedă, în numerar, depozite și titluri de stat pe termen scurt'),
      T('monthly public reports on the reserves; no interest paid to holders', 'rapoarte publice lunare despre rezerve; fără dobîndă plătită deținătorilor')]),
    (T('Common points of MiCA and GENIUS: full reserves, redemption at par, supervision of issuers', 'Puncte comune ale MiCA și GENIUS: rezerve complete, răscumpărare la paritate, supravegherea emitenților'),
     [T('they address the run on reserves of March 2023 and the algorithmic collapse of May 2022', 'răspund retragerii masive din martie 2023 și prăbușirii algoritmice din mai 2022')]),
    T('A fully reserved stablecoin is close to a narrow bank: safe, but its issuer earns the interest on the reserves', 'Un stablecoin cu rezerve complete este apropiat de o bancă îngustă (narrow bank): sigur, dar emitentul cîștigă dobînda la rezerve')),
    ph('capitol', T('United States Capitol, Washington, D.C.', 'Capitoliul SUA, Washington, D.C.'), h='0.24\\textheight'), wl='0.62', wr='0.35'), 'footnotesize')

D.recap(('Stablecoins', 'stablecoins'), [
    T('Three types: fiat-backed, crypto-backed, algorithmic; a supply of @{sup.total_end} billion USD in September 2026', 'Trei tipuri: garantate fiat, garantate cripto, algoritmice; o ofertă de @{sup.total_end} de miliarde USD în septembrie 2026'),
    T('Peg deviation in bp: $10\\,000(P - 1)$; small and short-lived, except in crises', 'Abaterea de la paritate în bp: $10\\,000(P - 1)$; mică și de scurtă durată, cu excepția crizelor'),
    T('USDC (2023): a run on the reserves, rescued; UST (2022): no reserves, collapse; MiCA and GENIUS require full reserves', 'USDC (2023): o retragere masivă din cauza rezervelor, salvată; UST (2022): fără rezerve, prăbușire; MiCA și GENIUS cer rezerve complete')])

# =============================================================================
# 9. RISCUL UNEI POZIȚII CRIPTO
# =============================================================================
D.section('The Risk of a Crypto Position', 'Riscul unei poziții cripto')

D.frame(T('VaR 1\\% and ES 2.5\\% in Dollars', 'VaR 1\\% și ES 2,5\\% în dolari'), items(
    (T('Chapter 10: $\\mathrm{VaR}_\\alpha = -q_\\alpha$, the loss exceeded with probability $\\alpha$ (VaR 1\\%); $\\mathrm{ES}_\\alpha = -E[r \\mid r \\le q_\\alpha]$ (ES 2.5\\%)', 'Capitolul 10: $\\mathrm{VaR}_\\alpha = -q_\\alpha$, pierderea depășită cu probabilitatea $\\alpha$ (VaR 1\\%); $\\mathrm{ES}_\\alpha = -E[r \\mid r \\le q_\\alpha]$ (ES 2,5\\%)'),
     [T('$q_\\alpha$: the $\\alpha$-quantile of the daily return $r$; $E[r \\mid r \\le q_\\alpha]$: the mean return on the days beyond the quantile', '$q_\\alpha$: cuantila de ordin $\\alpha$ a randamentului zilnic $r$; $E[r \\mid r \\le q_\\alpha]$: randamentul mediu în zilele de dincolo de cuantilă'),
      T('historical simulation (HS), Normal, Student-t (maximum likelihood), GARCH(1,1)-t for the next day', 'simularea istorică (HS), distribuția Normală, Student-t (verosimilitate maximă), GARCH(1,1)-t pentru ziua următoare')]),
    (T('From log returns to dollars: a log return of $-v\\%$ means a loss of $W(1 - e^{-v/100})$ on a position of value $W$', 'De la randamente logaritmice la dolari: un randament logaritmic de $-v\\%$ înseamnă o pierdere $W(1 - e^{-v/100})$ pentru o poziție cu valoarea $W$'),
     [T('Bitcoin, HS: VaR 1\\% $= @{v.btc.hs_v}\\%$; $W = 100\\,000$ USD: $100\\,000 \\times (1 - @{vx.e}) = @{v.btc.hs_v.usd}$ USD', 'Bitcoin, HS: VaR 1\\% $= @{v.btc.hs_v}\\%$; $W = 100\\,000$ USD: $100\\,000 \\times (1 - @{vx.e}) = @{v.btc.hs_v.usd}$ USD')]),
    T('Horizon: crypto trades every day, so a 10-day VaR covers 10 calendar days, not two trading weeks', 'Orizontul: activele cripto se tranzacționează zilnic, deci un VaR pe 10 zile acoperă 10 zile calendaristice, nu două săptămîni de tranzacționare')))

RK = ['btc', 'eth', 'sp500']
D.frame(T('Four Methods, Three Assets', 'Patru metode, trei active'), table(
    'lrrrrrrrr', T('& \\multicolumn{4}{c}{VaR 1\\% (\\%)} & \\multicolumn{4}{c}{ES 2.5\\% (\\%)} \\\\ & HS & Normal & t & GARCH-t & HS & Normal & t & GARCH-t',
                   '& \\multicolumn{4}{c}{VaR 1\\% (\\%)} & \\multicolumn{4}{c}{ES 2,5\\% (\\%)} \\\\ & HS & Normală & t & GARCH-t & HS & Normală & t & GARCH-t'),
    [f'{NM[k]} & ' + ' & '.join(f'$@{{v.{k}.{c}}}$' for c in ['hs_v', 'n_v', 't_v', 'g_v', 'hs_e', 'n_e', 't_e', 'g_e']) for k in RK],
    size='scriptsize') + table(
    'lrrrr', T('Position of 100\\,000 USD & VaR 1\\% HS & ES 2.5\\% HS & VaR 1\\% Normal & 10-day VaR 1\\% HS', 'Poziție de 100\\,000 USD & VaR 1\\% HS & ES 2,5\\% HS & VaR 1\\% Normal & VaR 1\\% HS pe 10 zile'),
    [f'{NM[k]} & @{{v.{k}.hs_v.usd}} & @{{v.{k}.hs_e.usd}} & @{{v.{k}.n_v.usd}} & @{{v.{k}.hs10.usd}}' for k in RK], size='scriptsize') + items(
    T('Whole samples until @{end}; t: Student-t by maximum likelihood ($\\hat\\nu = @{v.btc.nu}$ for Bitcoin); GARCH-t: the forecast for the next day ($\\hat\\sigma = @{v.btc.g_sigma}\\%$)', 'Eșantioanele complete pînă pe @{end}; t: Student-t prin verosimilitate maximă ($\\hat\\nu = @{v.btc.nu}$ pentru Bitcoin); GARCH-t: prognoza pentru ziua următoare ($\\hat\\sigma = @{v.btc.g_sigma}\\%$)')), 'scriptsize')

D.frame(T('Interpretation of the Risk Numbers', 'Interpretarea cifrelor de risc'), items(
    (T('A Bitcoin position is about three times as risky as an S\\&P 500 position: VaR 1\\% @{v.btc.hs_v}\\% against @{v.sp500.hs_v}\\%', 'O poziție în Bitcoin este de circa trei ori mai riscantă decît una în S\\&P 500: VaR 1\\% @{v.btc.hs_v}\\% față de @{v.sp500.hs_v}\\%'),
     [T('the Normal VaR 1\\% is @{v.btc.gap}\\% below the historical one for Bitcoin: heavy tails again', 'VaR 1\\% Normal este cu @{v.btc.gap}\\% sub cel istoric pentru Bitcoin: din nou cozile groase')]),
    (T('Conditional against unconditional: GARCH-t gives @{v.btc.g_v}\\% today, below the historical @{v.btc.hs_v}\\%, because volatility is now low', 'Condiționat față de necondiționat: GARCH-t dă azi @{v.btc.g_v}\\%, sub valoarea istorică de @{v.btc.hs_v}\\%, deoarece volatilitatea este acum scăzută'),
     [T('the historical number is a long-run average; the GARCH number is today\'s risk', 'cifra istorică este o medie pe termen lung; cifra GARCH este riscul de azi')]),
    (T('10 days: the historical VaR 1\\% of 10-day returns is @{v.btc.hs10}\\%; the square-root-of-time rule gives @{v.btc.sqrt10}\\%', '10 zile: VaR 1\\% istoric al randamentelor pe 10 zile este @{v.btc.hs10}\\%; regula rădăcinii pătrate a timpului dă @{v.btc.sqrt10}\\%'),
     [T('the square-root-of-time rule: $\\mathrm{VaR}(h\\text{ days}) = \\sqrt{h}\\,\\mathrm{VaR}(1\\text{ day})$, exact only for i.i.d.\\ Normal returns', 'regula rădăcinii pătrate a timpului: $\\mathrm{VaR}(h\\text{ zile}) = \\sqrt{h}\\,\\mathrm{VaR}(1\\text{ zi})$, exactă doar pentru randamente Normale i.i.d.'),
      T('the rule overstates: with tail index $\\alpha > 2$, an extreme quantile of a sum of $h$ returns grows roughly like $h^{1/\\alpha}$, more slowly than $\\sqrt{h}$', 'regula supraestimează: cu tail index-ul $\\alpha > 2$, o cuantilă extremă a sumei a $h$ randamente crește aproximativ ca $h^{1/\\alpha}$, mai lent decît $\\sqrt{h}$')]),
    T('ES 2.5\\% (HS) is close to VaR 1\\% (HS) for Bitcoin, as in Chapter 10 for equities', 'ES 2,5\\% (HS) este apropiat de VaR 1\\% (HS) pentru Bitcoin, ca în Capitolul 10 pentru acțiuni')))

chart(T('Backtesting Bitcoin VaR 1\\%', 'Backtesting pentru VaR 1\\% al Bitcoin'), 'sfm_ch14_btc_var', 'SFM_ch14_crypto_var', [
    T('One-day forecasts since @{bt.start}: HS on the last 365 days; GARCH(1,1)-t re-estimated every 365 days on all earlier data', 'Prognoze pe o zi din @{bt.start}: HS pe ultimele 365 de zile; GARCH(1,1)-t reestimat la fiecare 365 de zile pe toate datele anterioare'),
    T('Exceptions (expected @{bt.hs.exp}): HS @{bt.hs.x} (@{bt.hs.rate}\\%), GARCH-t @{bt.g.x} (@{bt.g.rate}\\%); Kupiec \\refKupiec: $LR = @{bt.hs.lr}$ (p $= @{bt.hs.p}$) and $@{bt.g.lr}$ (p $= @{bt.g.p}$); $LR \\sim \\chi^2(1)$ if the exception rate is 1\\%',
      'Depășiri (așteptat @{bt.hs.exp}): HS @{bt.hs.x} (@{bt.hs.rate}\\%), GARCH-t @{bt.g.x} (@{bt.g.rate}\\%); Kupiec \\refKupiec: $LR = @{bt.hs.lr}$ (p $= @{bt.hs.p}$) și $@{bt.g.lr}$ (p $= @{bt.g.p}$); $LR \\sim \\chi^2(1)$ dacă rata depășirilor este 1\\%'),
    T('Interpretation: both pass the Kupiec test; HS needs a higher VaR on average (@{bt.hs.mv}\\% against @{bt.g.mv}\\%), GARCH-t reacts faster but is exceeded more often (2018: @{bt.g.y2018} exceptions)',
      'Interpretare: ambele trec testul Kupiec; HS cere în medie un VaR mai mare (@{bt.hs.mv}\\% față de @{bt.g.mv}\\%), GARCH-t reacționează mai repede, dar este depășit mai des (2018: @{bt.g.y2018} depășiri)')],
    h='0.44\\textheight')

D.recap(('The Risk of a Crypto Position', 'riscul unei poziții cripto'), [
    T('Convert log-return VaR to money with $W(1 - e^{-v/100})$', 'Convertim VaR din randamente logaritmice în bani cu $W(1 - e^{-v/100})$'),
    T('Bitcoin VaR 1\\% is about three times that of the S\\&P 500; the Normal understates it', 'VaR 1\\% al Bitcoin este de circa trei ori cel al S\\&P 500; distribuția Normală îl subestimează'),
    T('Historical simulation and GARCH-t both pass the backtest since 2018', 'Simularea istorică și GARCH-t trec amîndouă backtesting-ul din 2018')])

# =============================================================================
# 10. AI
# =============================================================================
D.section('AI for Scientific Discovery', 'AI pentru descoperire științifică')

D.frame(T('An Open Question', 'O întrebare deschisă'), items(
    (T('\\textbf{Do stablecoin flows move crypto prices?}', '\\textbf{Influențează fluxurile de stablecoins prețurile activelor cripto?}'),
     [T('\\refGS found that Tether issuance supported Bitcoin in 2017; the stablecoin market is now about @{sup.growth_x} times larger than in 2020', '\\refGS au găsit că emisiunea de Tether a susținut Bitcoin în 2017; piața stablecoin-urilor este acum de circa @{sup.growth_x} de ori mai mare decît în 2020'),
      T('starter result: weekly log change of the stablecoin supply against Bitcoin\'s return, @{ai.n} weeks since 2020', 'rezultat de pornire: variația logaritmică săptămînală a ofertei de stablecoins față de randamentul Bitcoin, @{ai.n} săptămîni din 2020')]),
    (T('Same week: correlation @{ai.corr_same}; next week: @{ai.corr_next}, slope @{ai.b} ($t = @{ai.t}$, p $= @{ai.p}$, $R^2 = @{ai.r2}\\%$)', 'Aceeași săptămînă: corelația @{ai.corr_same}; săptămîna următoare: @{ai.corr_next}, panta @{ai.b} ($t = @{ai.t}$, p $= @{ai.p}$, $R^2 = @{ai.r2}\\%$)'),
     [T('flows and prices move together; flows do not predict the next week: cause, effect or common demand?', 'fluxurile și prețurile se mișcă împreună; fluxurile nu prognozează săptămîna următoare: cauză, efect sau cerere comună?')]),
    T('AI tools can speed up such a study; they do not replace checking it \\refWang', 'Instrumentele AI pot accelera un astfel de studiu; nu înlocuiesc verificarea lui \\refWang')))

D.frame(T('How AI Could Help', 'Contribuția posibilă a AI'), items(
    T('\\textbf{Literature}: a list of studies on stablecoin issuance and crypto prices, with their data and periods', '\\textbf{Literatura}: o listă a studiilor despre emisiunea de stablecoins și prețurile cripto, cu datele și perioadele lor'),
    T('\\textbf{Data}: code that downloads the supply of each stablecoin from DefiLlama and aligns it with daily prices', '\\textbf{Date}: cod care descarcă oferta fiecărui stablecoin de la DefiLlama și o aliniază cu prețurile zilnice'),
    T('\\textbf{Robustness}: daily and weekly data, USDT and USDC separately, sub-periods, Granger causality in both directions', '\\textbf{Robustețe}: date zilnice și săptămînale, USDT și USDC separat, subperioade, cauzalitate Granger în ambele direcții'),
    (T('Example prompt', 'Exemplu de prompt'),
     [T('\\aiprompt{Write Python code that downloads the daily supply of USDT and USDC from the DefiLlama stablecoins API, computes weekly log changes, joins them with weekly Bitcoin log returns, and tests whether supply changes Granger-cause returns and returns Granger-cause supply changes, with HAC standard errors.}',
        '\\aiprompt{Write Python code that downloads the daily supply of USDT and USDC from the DefiLlama stablecoins API, computes weekly log changes, joins them with weekly Bitcoin log returns, and tests whether supply changes Granger-cause returns and returns Granger-cause supply changes, with HAC standard errors.}')])))

D.frame(T('What to Check', 'Verificări necesare'), items(
    T('Annualisation: 365 days for crypto, 252 for equities; never one number for both', 'Anualizarea: 365 de zile pentru cripto, 252 pentru acțiuni; niciodată o singură valoare pentru ambele'),
    T('Alignment: prices joined on common days before returns; the same week definition for both series', 'Alinierea: prețurile unite în zilele comune înainte de randamente; aceeași definiție a săptămînii pentru ambele serii'),
    T('Units: basis points against percent (285 bp $= 2.85\\%$); billions against millions in supply data', 'Unitățile: puncte de bază față de procente (285 bp $= 2{,}85\\%$); miliarde față de milioane în datele de ofertă'),
    T('VaR convention: VaR 1\\% is a positive loss; never ``VaR 99\\%\'\'', 'Convenția VaR: VaR 1\\% este o pierdere pozitivă; niciodată „VaR 99\\%”'),
    T('Look-ahead: the supply series must be known at the date of the forecast; revisions of the data provider count', 'Look-ahead: seria ofertei trebuie să fie cunoscută la data prognozei; revizuirile furnizorului de date contează'),
    T('References: every cited paper must exist; check the DOI', 'Referințele: fiecare lucrare citată trebuie să existe; verificați DOI-ul')))

D.frame(T('Project Idea', 'Idee de proiect'), items(
    (T('\\textbf{Question}: do weekly changes in USDT and USDC supply help to forecast Bitcoin\'s return or volatility, and did this change after the spot ETFs?',
       '\\textbf{Întrebarea}: ajută variațiile săptămînale ale ofertei de USDT și USDC la prognoza randamentului sau a volatilității Bitcoin și s-a schimbat acest lucru după ETF-urile spot?'),
     [T('data: DefiLlama supply (public, no key) and the course data for prices', 'date: oferta de la DefiLlama (publică, fără cheie) și datele cursului pentru prețuri')]),
    (T('Steps', 'Pași'),
     [T('weekly log changes; correlation and Granger tests in both directions with HAC standard errors', 'variații logaritmice săptămînale; corelația și testele Granger în ambele direcții, cu erori standard HAC'),
      T('out-of-sample forecasts of next week\'s return and of the absolute return, against a random walk', 'prognoze în afara eșantionului pentru randamentul și pentru randamentul absolut din săptămîna următoare, comparate cu un mers aleator'),
      T('sub-periods: before and after Terra (May 2022) and the spot ETFs (January 2024)', 'subperioade: înainte și după Terra (mai 2022) și ETF-urile spot (ianuarie 2024)')]),
    T('Deliverable: one table, one chart, and a paragraph on what the data can and cannot show', 'Livrabil: un tabel, un grafic și un paragraf despre ce pot și ce nu pot arăta datele'),
    T('Declare any AI use, and list the errors of the AI that you corrected', 'Declarați orice utilizare a instrumentelor AI și enumerați erorile acestora pe care le-ați corectat')))

# =============================================================================
# REZUMAT
# =============================================================================
D.section('Summary', 'Rezumat')

D.frame(T('Key Takeaways', 'Idei de reținut'), items(
    T('Bitcoin: a public ledger secured by proof of work, with a fixed supply of 21 million; Ethereum runs smart contracts under proof of stake', 'Bitcoin: un registru public securizat prin proof of work, cu o ofertă fixă de 21 de milioane; Ethereum execută contracte inteligente prin proof of stake'),
    T('Annualise with 365 days; crypto volatility is @{ratio.btc} to @{ratio.sol} times that of the S\\&P 500, with heavy tails and IGARCH-like clustering', 'Anualizăm cu 365 de zile; volatilitatea cripto este de @{ratio.btc} pînă la @{ratio.sol} ori cea a S\\&P 500, cu cozi groase și volatility clustering de tip IGARCH'),
    T('Robust VR tests do not reject weak-form efficiency since 2014; the weekend is calmer in variance, not different in mean', 'Testele VR robuste nu resping eficiența în formă slabă din 2014; weekendul este mai calm în varianță, nu diferit în medie'),
    T('Correlation with equities rose from about 0 to about @{c.btc_sp500.post2020} after 2020; Bitcoin is no safe haven', 'Corelația cu acțiunile a crescut de la circa 0 la circa @{c.btc_sp500.post2020} după 2020; Bitcoin nu este un activ de refugiu'),
    T('Drawdowns of more than 50\\% are regular; the ETF effect on volatility is not significant', 'Drawdown-urile de peste 50\\% sînt obișnuite; efectul ETF-urilor asupra volatilității nu este semnificativ'),
    T('Stablecoins hold a peg by arbitrage and reserves; USDC 2023 and UST 2022 show the two ways a peg breaks', 'Stablecoins își mențin paritatea prin arbitraj și rezerve; USDC în 2023 și UST în 2022 arată cele două moduri în care se rupe o paritate')))

D.frame(T('Key Formulas', 'Formule de reținut'), '{\\renewcommand{\\arraystretch}{1.4}' + table(
    'll', T('\\textbf{Quantity}', '\\textbf{Mărimea}') + ' & ' + T('\\textbf{Formula}', '\\textbf{Formula}'),
    [T('Bitcoin supply', 'Oferta de Bitcoin') + ' & $210\\,000 \\times 50 \\times \\sum_{j \\ge 0} 2^{-j} = 21$ ' + T('million', 'milioane'),
     T('Annualisation', 'Anualizarea') + ' & ' + T('mean $A\\bar r$, volatility $\\sqrt{A}\\,s$; $A = 365$ (crypto), $252$ (equities)', 'media $A\\bar r$, volatilitatea $\\sqrt{A}\\,s$; $A = 365$ (cripto), $252$ (acțiuni)'),
     'Hill & $\\hat\\alpha = \\big[\\frac1k\\sum_{i \\le k}\\ln(X_{(i)}/X_{(k+1)})\\big]^{-1}$',
     'GARCH(1,1) & $\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$, ' + T('half-life', 'înjumătățire') + ' $\\frac{\\ln 0.5}{\\ln(\\alpha + \\beta)}$',
     T('Variance ratio', 'Raportul varianțelor') + ' & $\\mathrm{VR}(q) = \\mathrm{Var}(r_t + \\dots + r_{t-q+1})/(q\\,\\mathrm{Var}(r_t))$',
     'Drawdown & $D_t = P_t/\\max_{s \\le t}P_s - 1$; ' + T('recovery gain', 'cîștigul de recuperare') + ' $d/(1 - d)$',
     T('Peg deviation', 'Abaterea de la paritate') + ' & $d_t = 10\\,000\\,(P_t - 1)$ bp; ' + T('half-life', 'înjumătățire') + ' $\\ln 0.5/\\ln\\phi$',
     T('VaR in money', 'VaR în bani') + ' & $W(1 - e^{-\\mathrm{VaR}/100})$, $\\mathrm{VaR}_\\alpha = -q_\\alpha$'],
    size='scriptsize') + '}')

D.frame(T('Check Yourself', 'Autoevaluare'), items(
    (T('\\textbf{Question}: a fund reports a Bitcoin volatility of 55\\% a year from a daily standard deviation of 3.5\\%. Is it right?', '\\textbf{Întrebare}: un fond raportează o volatilitate anuală a Bitcoin de 55\\% pornind de la o abatere standard zilnică de 3,5\\%. Este corect?'),
     [T('\\textbf{Answer}: no: $3.5 \\times \\sqrt{252} \\approx 55.6\\%$; with 365 days it is $3.5 \\times \\sqrt{365} \\approx 66.9\\%$', '\\textbf{Răspuns}: nu: $3{,}5 \\times \\sqrt{252} \\approx 55{,}6\\%$; cu 365 de zile se obține $3{,}5 \\times \\sqrt{365} \\approx 66{,}9\\%$')]),
    (T('\\textbf{Question}: USDT closes at 1.0040 USD. What is the peg deviation?', '\\textbf{Întrebare}: USDT se închide la 1,0040 USD. Cît este abaterea de la paritate?'),
     [T('\\textbf{Answer}: $10\\,000 \\times 0.0040 = 40$ bp above the peg: demand for USDT exceeds what arbitrage has supplied', '\\textbf{Răspuns}: $10\\,000 \\times 0{,}0040 = 40$ bp peste paritate: cererea de USDT depășește ce a furnizat arbitrajul')]),
    (T('\\textbf{Question}: why was UST lost while USDC returned to the peg?', '\\textbf{Întrebare}: de ce s-a pierdut UST, în timp ce USDC a revenit la paritate?'),
     [T('\\textbf{Answer}: USDC had dollar reserves whose bank deposits were guaranteed; UST was backed only by LUNA, whose value fell with UST', '\\textbf{Răspuns}: USDC avea rezerve în dolari, ale căror depozite bancare au fost garantate; UST era garantat doar de LUNA, a cărei valoare a scăzut odată cu UST')]),
    T('Further reading by the author of the course: \\refPeleA (are cryptos alternative assets?) and \\refPeleB (Bitcoin VaR)', 'Lecturi suplimentare ale autorului cursului: \\refPeleA (sînt activele cripto active alternative?) și \\refPeleB (VaR pentru Bitcoin)'),
    T('Next: Chapter 15, systemic risk', 'Urmează: Capitolul 15, riscul sistemic')))

D.references(BIB, per=13)

if __name__ == '__main__':
    D.write(V)
