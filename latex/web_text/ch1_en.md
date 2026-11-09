<!--
Course text of the web version, Chapter 1 (EN). Inserted into the page by latex/build_web_lecture.py.
Format: # section title (introduction), ## slide title {h=hash} (text), ## @end (summary).
Numbered equations: $$ ... $$ {#eq:label}, reference @eq:label. Citations: @Key -> Author (year), [@Key] -> (Author, year).
Current hashes: python3 build_web_lecture.py 1 --hashes
-->

## @intro

@text
A price level on its own says little about an investment. A share quoted at a few lei and an index quoted in thousands of points cannot be compared, and neither number shows how much an investor gained, how much the value fluctuated along the way or how deep the worst fall was. This chapter turns raw market data into numbers that do allow such comparisons: returns, measures of risk and drawdowns, together with their estimation error.

The chapter works throughout with real daily data for the BET index of the Bucharest Stock Exchange (BVB), the S&P 500, the DAX, Bitcoin and five BVB stocks, over the comparison period 2015–2026. The data already show the main themes of the course. Over this period 100 units invested in Bitcoin grew to 29,475, against 379 in the S&P 500, but Bitcoin's maximum drawdown was −83.4%. Daily returns have heavy tails: beyond three standard deviations the S&P 500 has 5.2 times as many days as the Normal distribution predicts. The annualised Sharpe ratio of the S&P 500, 0.72, has a standard error of 0.29, so most rankings of assets by such indicators are not statistically significant.

Before any statistic, the chapter deals with the quality of the data: the choice of the primary source, the adjustment of prices for dividends and splits, and the errors that real series contain.

@chart sfm_ch1_growth
Value of 100 units invested on the first day of 2015 in the S&P 500, the DAX, the BET and Bitcoin, on a logarithmic vertical axis: equal vertical distances correspond to equal percentage changes.

@goals
- **Choose** a reliable data source and check a price series before using it.
- **Compute** simple and log returns and aggregate them over time and across assets.
- **Annualise** means and volatilities with the actual frequency of the data.
- **Describe** a return series by its moments, quantiles and extreme values.
- **Compute and interpret** the CAGR, the Sharpe and Sortino ratios, the maximum drawdown, the Calmar ratio and volatility drag.
- **Compare** the difference between two indicators with their estimation error.

The student version of the seminar is available as a [PDF](../EN/Seminars/seminar1_data_returns_indicators.pdf).

@plan
- Financial Data: Sources and Quality | Types and frequencies of data, sources, adjusted prices, errors in real data and three biases.
- From Prices to Returns | Why statistics works with returns, weak stationarity and the ADF test.
- Simple and Log Returns | Definitions, the gap between the two returns on extreme days and when to use each.
- Multi-Period and Portfolio Returns | Aggregation over time, cumulative wealth and portfolio returns.
- Annualisation | Annual mean and volatility, the CAGR and the precision of the estimated mean.
- Descriptive Statistics of Returns | Location, spread, skewness and kurtosis, heavy tails and time-varying volatility.
- Performance Indicators | Sharpe, Sortino, drawdown, Calmar and volatility drag, with their estimation error.
- AI for Scientific Discovery | An open question on the persistence of Sharpe ratios and the checks an AI-assisted study needs.
- Summary | The formulas of the chapter and self-assessment questions.

# Financial Data: Sources and Quality

Every result in this course starts from a data file, and every error in that file passes into the results. This section describes what financial data are, how often they are recorded, where they come from and which data the course uses. It then shows, on real series, the two operations that most often decide whether a price series can be used: the adjustment for corporate actions and the check against a second source. It ends with three systematic biases that leave no visible trace in the data but make results look better than they are.

## What Financial Data Are {h=ba995ca6}

Financial data fall into four groups. **Market data** are the information produced by a trading venue: trade prices, the best bid and ask quotes, traded volume, and derived series such as index levels, exchange rates, interest rates and crypto prices. The best bid is the highest price at which someone is willing to buy at a given moment, the best ask the lowest price at which someone is willing to sell; the gap between them, the spread, is the cost of an immediate round trip.

**Corporate actions** form a separate group because they change the quoted price without changing the value of an investor's holding. When a company pays a cash dividend, the share price falls on the ex-dividend day by roughly the amount paid, but the shareholder receives the cash. In a split, each old share becomes several new ones at a proportionally lower price; in a consolidation (reverse split) several old shares become one. Bonus shares work like a split. A series of quoted prices that ignores these events contains jumps that no investor experienced.

**Fundamental and macroeconomic data**, such as balance sheets, earnings, inflation, gross domestic product (GDP) and policy rates, describe the firms and the economy behind the prices. **Alternative data** include news, social media, satellite images and blockchain transactions. The course works mainly with daily market data.

## Ticker Tape and Electronic Data Feeds {h=180f0ffa}

The first real-time price feed was the stock ticker, in use from 1867. The machine received trade prices by telegraph and printed them on a paper tape: the price of the last trade, a few minutes after it had taken place. Investors far from the exchange floor could follow the market almost as it happened.

Today exchanges publish electronic feeds that data vendors collect and redistribute. Every quote and every trade is recorded with a time stamp to the microsecond. Most statistical studies, including this course, do not use these records directly but their daily summaries, the OHLCV bars (open, high, low, close, volume) introduced [below](#s11).

The tape survives in two forms. Moving bands of prices and news still run along building facades, such as the Reuters building in New York shown on the slide. The short ticker symbols that identify securities also come from the tape: at the BVB, TLV is Banca Transilvania, SNP is OMV Petrom and H2O is Hidroelectrica.

## Data Frequency {h=406e301f}

The finest level of data is **tick data**: every trade and every quote. A liquid market produces millions of records per day. Tick prices carry microstructure noise; the best-known source is the bid–ask bounce, in which successive trades execute alternately at the bid and at the ask, so that the trade price jumps back and forth even when the value of the asset does not change. **Intraday bars**, of 1, 5 or 60 minutes, summarise the ticks and are the input of realised volatility, the subject of [Chapter 8](ch8.html).

**Daily data**, with one row per trading day, are the standard of this course and of most academic studies. Exchanges trade about 250 days per year, crypto markets 365 days per year, weekends and holidays included. **Weekly, monthly and annual data** give fewer observations but are closer to the horizon of a long-term investor.

The frequency matters for every annualised figure in this chapter. The rule is to annualise with the **actual** number of observations per year of each series, about 252 for an exchange and 365 for Bitcoin, never with a single factor for all assets. The consequences of using the wrong factor are shown in the section on [annualisation](#s38).

## Where the Data Come From {h=97631c6c}

The table on the slide lists six types of source. Exchanges (the BVB, Deutsche Börse, the New York Stock Exchange, NYSE) publish official prices, volumes and index levels, on their sites and through paid feeds. Data vendors such as Bloomberg, LSEG (the London Stock Exchange Group, formerly Refinitiv) and EODHD (EOD Historical Data, where EOD means end of day) gather prices from many markets, compute adjusted prices and add fundamental data, by subscription. Central banks, such as the National Bank of Romania (BNR), the European Central Bank (ECB) and the Federal Reserve, publish reference exchange rates and interest rates free of charge.

Statistical databases such as FRED (Federal Reserve Economic Data, from the Federal Reserve Bank of St. Louis) and Eurostat provide macroeconomic series and government bond yields. Academic databases such as CRSP (the Center for Research in Security Prices at the University of Chicago) and the Kenneth French data library provide cleaned returns, including those of delisted firms, and factor returns, under a university licence or free. Crypto exchanges and aggregators supply prices around the clock and on-chain data.

A vendor collects and cleans the data, but it is not the origin of the number. When two sources disagree, the reference is the **primary source** of the figure: the exchange for a closing price, the central bank for a reference exchange rate. The [EUR/RON example](#s16) shows why this rule matters.

## The Data of This Course {h=3e6765e7}

The course uses daily market data from EODHD, stored as a fixed snapshot so that every result can be reproduced. The collection holds 91 series: indices, exchange-traded funds (ETFs, funds traded on an exchange that track an index), stocks from the BVB, the US and Europe, crypto assets, exchange rates and bond yields. Each series has the same columns: date, open, high, low, close, adjusted close and volume.

The EUR/RON exchange rate is the exception. The course uses the official BNR reference rate, because the EODHD series for this rate contains erroneous quotes; the frame [Data Errors: Two EUR/RON Series](#s16) measures the damage.

Three conventions apply throughout. Indices, exchange rates and crypto assets use the close; stocks and ETFs use the adjusted close, so that dividends and splits are taken into account. Weekend quotes and repeated closes on holidays are removed, except for crypto assets, which trade every day. Each series is analysed on its own calendar, so the BET and the S&P 500 keep their own trading days. The comparison period of the chapter runs from 1 January 2015 to 18 September 2026.

## Daily Bars: OHLCV {h=d8b4bd7f}

A daily bar records five numbers for day $t$: the opening price $O_t$, the highest price $H_t$, the lowest price $L_t$, the closing price $C_t$ and the volume $V_t$, the number of shares traded. By construction, the open and the close lie between the low and the high, $L_t \le O_t, C_t \le H_t$. The range $H_t - L_t$ measures how far the price moved during the day and is the basis of the range-based volatility estimators of [Chapter 8](ch8.html).

The table shows the last five trading days of Banca Transilvania (TLV) in the course data, from 14 to 18 September 2026. Four of the rows look ordinary: the close moves between 33.40 and 35.50 lei and the volume between 346,819 and 1,884,988 shares. The bar of 17 September 2026 is different. Open, high, low and close are all equal to 33.40 lei, the same value as the close of the previous day, while the volume is 1,084,380 shares.

A flat bar is normal for an illiquid share on a day with a single trade, or for a holiday filled in with the previous price. With more than a million shares traded, a single price for the whole day is very unlikely, so the bar is suspect and has to be checked against the exchange. Simple checks of this kind, the inequalities between the four prices, flat bars and zero volumes, find many data errors before they reach any statistic.

## Corporate Actions and the Adjusted Close {h=75c2b618}

On the day of a corporate action the closing price $P_t$ jumps, although the value of the holding does not change. The adjusted price removes these jumps. Each event $k$, on day $t_k$, has an **adjustment factor** $F_{t_k}$. For a cash dividend $D$, $F = 1 - D/P_{t-1}$: the price falls by about $D$ on the ex-dividend day, and scaling the earlier prices by $F$ removes this fall. For an $m$-for-$n$ split, in which $n$ old shares become $m$ new ones, $F = n/m$; a consolidation is a split with $m < n$, so its factor is greater than one and scales the earlier prices up.

The adjusted price multiplies each past price by the factors of all later events:

$$ P^{\text{adj}}_t = P_t \prod_{k:\, t_k > t} F_{t_k}. $$ {#eq:adj}

The product in @eq:adj runs only over events after day $t$, so the latest prices remain unchanged and the adjustment works backwards in time. Returns computed from adjusted prices include the dividend and contain no artificial jumps.

The same idea appears directly in the definition of the return. With a dividend $D_t$ paid on day $t$, the simple return is $R_t = (P_t + D_t)/P_{t-1} - 1$ [@FrankeHardleHafner2019]: the investor's gain is the change in price plus the cash received. A return computed from the adjusted close gives, up to a negligible second-order term, the same value.

## Worked Example: Adjusted Prices {h=e94f5bd4}

Consider a share quoted at 100 that pays a dividend of 2 on day 5 and has a 2-for-1 split on day 10. The dividend factor is $F_5 = 1 - 2/100 = 0.98$ and the split factor $F_{10} = 1/2 = 0.50$. A price before day 5 is followed by both events, so it is multiplied by $0.98 \times 0.50 = 0.49$, and 100 becomes 49. A price between days 5 and 9 is followed only by the split and is multiplied by 0.50. From day 10 onwards no event follows, and the adjusted price equals the quoted price.

The example shows a property of the adjusted series that is easy to forget: it changes every time a new event occurs. After the next dividend, every earlier adjusted price is scaled again. Two downloads of the same adjusted series made at different dates therefore differ, and a result can be reproduced only if the download date is kept together with the data.

Indices face the same choice. A **price index** ignores dividends, so its level falls when the constituents go ex-dividend. A **total return index** reinvests the dividends in the index and measures what an investor holding the constituents would have earned. The [BET and BET-TR comparison](#s15) shows how large the difference becomes over a decade.

## Real Data: Banca Transilvania, Close vs Adjusted Close {h=22fbbb02}

The chart compares the two prices of Banca Transilvania on a logarithmic vertical axis: the close as traded (red) and the adjusted close (green). On 16 August 2022 the traded price jumps from 2.12 to 21.20 lei. Nothing happened to the value of the bank: it was a consolidation of the shares, ten old shares into one new share.

A log return computed from the traded close records this day as a gain of +230% in a single day, a number that would dominate any mean, variance or kurtosis of the series. Computed from the adjusted close, the return of the same day is 0.0%. Before August 2022 the adjusted price lies above the traded one, because the consolidation factor of 10 scales all earlier prices up; dividends and bonus shares, which scale earlier prices down, narrow the gap. At the end of the sample the two lines coincide, as @eq:adj requires.

The consolidation is the most visible event but not the only one. Since 2015 the daily returns of the two series differ by more than 2 percentage points on 24 days, the days of dividends and bonus shares. Each of these days would add a spurious loss to a return series built from the traded price.

## The Effect of Dividends: BET vs BET-TR {h=b72c06e6}

The BET (Bucharest Exchange Trading) index is the benchmark index of the BVB and a price index. The BET-TR (BET Total Return) has the same constituents but reinvests their dividends. The chart follows 100 units invested in each on 23 September 2014: at the end of the sample the investment is worth 455 in the BET and 1,005 in the BET-TR. The two lines start together and separate a little more each year, because each dividend reinvested in the BET-TR earns the later returns as well.

In annual terms, the growth rate is 13.5% per year for the BET and 21.2% for the BET-TR. Dividends add 7.8 percentage points per year, more than half of the growth of the price index. On the BVB, ignoring dividends gives a very incomplete picture of what an investor in the market earned.

The lesson applies to every comparison of indices: compare like with like. The DAX is a total return index, while the S&P 500 and the BET are price indices. A comparison of the DAX with the S&P 500 or with the BET therefore favours the DAX by the dividend yield of the other two markets. The tables later in the chapter include the BET-TR next to the BET for this reason.

## Data Errors: Two EUR/RON Series {h=bbf2bded}

The chart plots the daily log returns of two series of the same exchange rate, EUR/RON: the series from EODHD (red) and the BNR reference rate (green). The two series have 5,154 days in common. On 412 of them, the daily change of the EODHD series differs from that of the BNR rate by more than 1 percentage point, a very large difference for an exchange rate whose genuine daily changes are usually a small fraction of a percent. The red series shows a block of alternating erroneous quotes between about 2010 and 2014 and isolated spikes of more than 10% in both directions.

The errors change every statistic. The annualised volatility of the EODHD series is 13.4%, against 4.5% for the BNR rate. The largest daily change of the EODHD series, in absolute value, is 15.4%, on 14 August 2025: it is the return to the correct level after a wrong quote on the previous day. A single bad quote thus produces two extreme returns of opposite sign, the error and its reversal, and both inflate the volatility and the kurtosis.

The solution is the rule of the [sources frame](#s9): use the primary source of the figure. For a reference exchange rate the primary source is the central bank, and the course uses the BNR series. An error of this kind is found by comparing two sources, or by listing the largest returns and checking each of them.

## Three Biases That Overstate Performance {h=c3fe0580}

Some errors do not lie in individual prices but in the way the sample is chosen or used. **Survivorship bias** arises when the sample contains only the firms or funds that still exist. Firms with poor results were delisted, merged or went bankrupt, so they are missing, and the average return of the survivors overstates what an investor could have earned [@BrownEtAl1992]. An example is the current composition of the BET traced back to 2000: the companies that left the index in the meantime are not in the sample.

**Look-ahead bias** is the use of information that was not available at the date of the decision. A typical case is an annual report published in March and used in a backtest for decisions taken in January. The backtest then trades on numbers the investor could not yet know. Database dates often record the period to which a figure refers, not the date of its publication, which makes this error easy to commit.

**Data snooping** is trying many rules on the same data and reporting the best one. Even if no rule works, the best of many looks significant by chance. @HarveyLiuZhu2016 conclude that hundreds of published predictors of returns are probably false discoveries. All three biases overstate returns, and none of them shows up in the descriptive statistics of the final sample, which look perfectly normal. They can only be prevented by the design of the study.

## A Data Checklist {h=544d02ac}

The errors of this section can be caught by a short routine applied to every new series. First, record the source: the primary source of the figure and the column used, close or adjusted close. Second, check the calendar: weekends, holidays and missing days, and, in a joint analysis of several assets, the same dates for all of them.

Third, look for jumps: list the 10 largest returns in absolute value and check each against the news of that day. A genuine crash has a known cause; a consolidation or a wrong quote does not. Fourth, look for flat bars, zero volume and repeated prices, which reveal holidays filled in with the last price or stale prices, quotes that were not updated. Fifth, check units and currency: lei, euro or dollars, index points, prices quoted in cents.

Finally, keep the raw file and the download date, and never edit the data by hand. Every correction is made in code, where it can be read, repeated and reversed. These rules cost little and protect every later result of the course.

## Recap: Data {h=4b01f1b4}

Prices come from exchanges, data vendors and central banks, and when sources disagree the primary source of the figure is the reference. For stocks the adjusted price must be used, and when dividends matter, as on the BVB, total return indices must be used instead of price indices: over 2014–2026 the BET-TR grew to 1,005 from an initial 100, the BET only to 455.

Real data contain errors, such as the ten-to-one consolidation of Banca Transilvania or the wrong quotes of the EODHD EUR/RON series. They are found by checking jumps, flat bars and a second source. Survivorship bias, look-ahead bias and data snooping overstate results without leaving any visible trace in the data.

## @end

A usable price series is the end of a chain of decisions: the source, the column, the calendar, the adjustment for corporate actions and the checks for errors. The examples of the section show that each link matters: one consolidation produces a fictitious one-day return of +230%, ignoring dividends removes 7.8 percentage points per year from the growth of the Romanian market, and a series with wrong quotes triples the volatility of an exchange rate. With clean prices in hand, the [next section](#sec-2) explains why the statistics of the course work with returns rather than prices.

# From Prices to Returns

The statistical methods of this course are applied to returns, not to prices. This section gives the reasons: returns have no unit, so they can be compared across assets, and they fluctuate around a stable level, while prices follow a trend. The second property is made precise by the notion of weak stationarity and checked formally with the augmented Dickey–Fuller (ADF) test on the S&P 500.

## Returns as the Object of Analysis {h=d0eb139d}

A price has a unit and a scale; a return has neither. A change of 1 leu is large for a share quoted at 2 lei and small for one quoted at 500 lei. A return expresses the change relative to the starting price, so returns of different assets, in different currencies and over different periods can be compared directly. The investor's question is also relative: by how much did my wealth change? That is exactly a return.

The second reason is statistical. Prices follow a trend and do not return to a fixed level, so their mean and variance depend on the period observed. Returns fluctuate around a stable level. The concept behind this distinction is **weak stationarity**: a series is weakly stationary if its mean is constant, its variance is constant and the covariance between two observations depends only on the distance between them, the lag, not on the date.

Most statistical methods of the course assume at least weak stationarity, because only then can averages over time estimate fixed population quantities. A sample mean of prices estimates nothing stable; a sample mean of returns estimates the expected return. The next two frames show the difference on data and test it formally.

## S&P 500: Price Level vs Daily Returns {h=6cfb19c3}

The upper panel shows the S&P 500 in index points since 2000. The price has an upward trend with long swings: the falls of 2000–2002 and 2008–2009 and a strong rise after 2012. Its mean and dispersion over any window depend on which years the window covers, which is what non-stationarity means in practice.

The lower panel shows the daily log returns of the same index. They fluctuate around zero over the whole period, in calm years and in turbulent ones, such as the financial crisis of 2008 and the pandemic shock of 2020. The level of the series does not drift.

The lower panel also shows that the size of the fluctuations is not the same at all times: large returns come in clusters, around 2008 and 2020. This phenomenon, volatility clustering, does not contradict weak stationarity, which concerns the unconditional variance. A series can have a constant long-run variance while its variance conditional on the recent past changes from day to day. The models that describe this are the subject of [Chapter 9](ch9.html).

## A Formal Check: the ADF Test (1/2) {h=555e1574}

The augmented Dickey–Fuller (ADF) test [@DickeyFuller1979] asks whether a series has a unit root, that is, whether it behaves like a random walk. The null hypothesis $H_0$ is a unit root, so the series is non-stationary; the alternative $H_1$ is that the series is stationary. The test regresses the daily change of the series on its level on the previous day:

$$ \Delta y_t = \alpha + \beta t + \gamma\, y_{t-1} + \sum_{j=1}^{p} \delta_j\, \Delta y_{t-j} + \varepsilon_t . $$ {#eq:adf}

In @eq:adf, $y_t$ is the series tested, the log price or the log return, and $\Delta y_t = y_t - y_{t-1}$ is its change from day $t-1$ to day $t$. The constant $\alpha$ and the linear trend $\beta t$ allow for drift and trend; the $p$ lagged changes, with coefficients $\delta_j$, absorb short-term autocorrelation, and $\varepsilon_t$ is the error. The key coefficient is $\gamma$. If $\gamma = 0$, today's change does not depend on yesterday's level: the series has no level to return to, which is a unit root. If $\gamma < 0$, a high level is followed on average by a fall and a low level by a rise, so the series is pulled back towards its level and is stationary.

The test rejects $H_0$ when the $t$ statistic of $\hat\gamma$ lies below a critical value. The critical values do not come from the Student distribution but from the Dickey–Fuller tables, because under $H_0$ the regressor $y_{t-1}$ is itself non-stationary and the $t$ statistic has a non-standard distribution, shifted to the left. The critical value also depends on whether the constant and the trend are included.

## A Formal Check: the ADF Test (2/2) {h=efa13bd7}

On daily S&P 500 data since 2000, the test gives two clear answers. For the log price, with a trend in the regression, the statistic is −2.08, the 5% critical value is −3.41 and the p-value is 0.56: the unit root is not rejected. For the log returns, the statistic is −14.3, the 5% critical value is −2.86 and the p-value is below 0.001: the unit root is rejected. The two critical values differ because the first regression includes a trend and the second only a constant.

The p-value is the probability, under $H_0$, of obtaining a statistic at least as negative as the one observed. A p-value of 0.56 means that a random walk would produce a statistic like −2.08 more than half of the time, so the data give no reason to doubt the unit root. A statistic of −14.3 is practically impossible under a unit root. With the usual threshold of 0.05, $H_0$ is rejected for returns and kept for prices.

The conclusion supports the choice of the course: prices behave like a random walk, returns are stationary, so the statistical models are built for returns. Failing to reject a unit root does not prove that one exists, but the two results together give a consistent picture. Unit roots and the random walk hypothesis are studied in detail in [Chapter 7](ch7.html).

## @end

Returns are preferred to prices for two reasons: they have no unit, so they can be compared across assets and periods, and they are stationary while prices are not. On the S&P 500 since 2000, the ADF test keeps the unit root for the log price (statistic −2.08, p-value 0.56) and rejects it for the log returns (statistic −14.3). There are, however, two ways to define a return, and the [next section](#sec-3) shows when they differ and which one to use.

# Simple and Log Returns

A return can be defined as a relative change of the price or as the change of its logarithm. The two definitions give almost the same number for a typical day and very different numbers for a crash or over a long horizon. This section defines both, measures the gap between them, illustrates it on the most extreme days of Bitcoin and the S&P 500, and gives a rule for choosing between them.

## The Simple Return {h=586bf2a5}

The **simple (net) return** from $t-1$ to $t$ is the relative change of the price [@FrankeHardleHafner2019; @Tsay2010]:

$$ R_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1 . $$ {#eq:simple}

The ratio $1 + R_t = P_t/P_{t-1}$ is the **gross return**: the value at $t$ of one unit invested at $t-1$. From here on, $P_t$ denotes the closing price of an index or the adjusted price of a stock, so that dividends and splits are already taken into account.

The simple return is bounded below: $R_t \ge -1$. The bound expresses limited liability: the price cannot fall below zero, so a shareholder cannot lose more than the amount invested. For example, with $P_{t-1} = 100$ and $P_t = 110$, $R_t = 10\%$. This is the number an investor sees on an account statement when the fund reports that it "gained 10%".

## The Log Return {h=a671b0fa}

The **log return**, also called the continuously compounded return, is the change of the logarithm of the price:

$$ r_t = \ln\frac{P_t}{P_{t-1}} = \ln P_t - \ln P_{t-1} = \ln(1 + R_t), \qquad R_t = e^{r_t} - 1 . $$ {#eq:log}

The name comes from compounding. One leu invested at a rate $r$ compounded $m$ times during the period grows to $(1 + r/m)^m$, and as $m \to \infty$ this tends to $e^{r}$. The log return is therefore the rate that, compounded continuously over the period, turns $P_{t-1}$ into $P_t$. With $P_{t-1} = 100$ and $P_t = 110$, $r_t = \ln 1.1 = 9.53\%$, below the simple return of 10% on the same prices.

Unlike the simple return, the log return is unbounded: $r_t \in (-\infty, +\infty)$, and a total loss, $P_t = 0$, corresponds to $r_t = -\infty$. Log returns are the default in the statistics of financial markets. The main reason, shown in the [next section](#sec-4), is that they add up over time, so that the return over a long period is the sum of the daily returns. Their range over the whole real line also suits statistical models better.

## The Difference between r and R {h=98810099}

The relation between the two returns follows from the Taylor expansion of the logarithm around zero:

$$ r = \ln(1 + R) = R - \frac{R^2}{2} + \frac{R^3}{3} - \dots $$ {#eq:taylor}

The logarithm is concave, so its graph lies below its tangent at zero, the line $r = R$. Hence $r < R$ for every $R \neq 0$, for gains as well as for losses. For small changes the first correction term, $-R^2/2$, gives almost the whole gap.

For a typical daily change of $R = 1\%$, $r = 0.995\%$, and the approximation $R - R^2/2$ also gives 0.995%. The gap $R - r$ is 0.5 basis points; a **basis point** (bp) is 0.01 percentage points. For a large change of $R = 10\%$, $r = 9.531\%$, while $R - R^2/2 = 9.500\%$, and the gap $R - r$ is 46.9 basis points. For daily data the two returns are practically equal; for large changes and long horizons they differ, and the gap grows with the square of the change.

## Simple vs Log Returns on Real Extremes {h=ccff5845}

The chart plots the log return against the simple return. The dashed line is the 45-degree line $r = R$, the solid curve is $r = \ln(1 + R)$, and the markers are the largest daily moves of Bitcoin (circles) and of the S&P 500 (squares). Near zero the curve and the line coincide; away from zero the curve lies below the line on both sides, and the gap grows much faster on the side of losses.

The largest daily moves in the course data show the size of the gap. On 12 March 2020 Bitcoin had a simple return of −37.2% but a log return of −46.5%. On 7 December 2017 it had a simple return of +25.2% and a log return of +22.5%. On 16 March 2020 the S&P 500 had a simple return of −12.0% and a log return of −12.8%.

Because the curve is concave, losses look larger and gains look smaller when measured in log returns. The asymmetry has a precise meaning: a log return of −46.5% and one of +46.5% cancel exactly, while a simple loss of 37.2% needs a simple gain of much more than 37.2% to be recovered. On days like these the choice of return changes the number reported by several percentage points.

## Worked Example: Up 10%, Then Down 10% {h=f14808b3}

Consider the price path $100 \to 110 \to 99$. The simple returns are $+10\%$ and $-10\%$, and their sum, zero, suggests that the investor is back where they started. The investor has in fact lost: compounding the two returns gives $1.1 \times 0.9 - 1 = -1.0\%$, the change from 100 to 99.

The log returns of the same path are $r_1 = \ln 1.1 = 9.53\%$ and $r_2 = \ln 0.9 = -10.54\%$. Their sum is $-1.01\%$, which equals $\ln(99/100)$ exactly. Adding log returns gives the correct two-period log return without any correction.

The rule that follows is the main practical difference between the two returns: simple returns compound, that is, gross returns multiply, while log returns add. A gain and a loss of the same size in percent do not cancel; in simple returns the loss of 10% applies to a larger base than the gain of 10%.

## Question for the Room: Up 50%, Down 50% {h=b723783b}

The question pushes the previous example to larger numbers. A share rises by 50% in one year and falls by 50% in the next. What does the price reach after two years, relative to its initial level?

Compounding the gross returns gives $1.5 \times 0.5 = 0.75$: wealth has changed by −25%, although the two percentages have the same size. In log returns, $\ln 1.5 + \ln 0.5 = \ln 0.75 = -28.8\%$, a direct sum.

The general form of the result is the recovery rule. After a loss $x$, wealth is $1 - x$ times its previous level, and returning to that level requires a gain of $x/(1 - x)$. A loss of 50% therefore requires a gain of 100%. The required gain grows faster than the loss, which is why the depth of a fall matters so much in the [drawdown measures](#s55) of the chapter.

## Choosing the Type of Return {h=d9e3f848}

The table on the slide matches each situation with a return. For aggregation over time, log returns are used, because the $k$-period log return is a sum, $r_t(k) = r_t + \dots + r_{t-k+1}$. For a portfolio of several assets, simple returns are used, because the portfolio return is exactly the weighted average $R_p = \sum_i w_i R_i$. For statistical modelling, log returns are preferred, because they are unbounded and their distribution is closer to symmetric.

For reporting to investors, the simple return is used, since "the fund gained 10%" is a simple return. For daily data and small changes either return will do, because the difference is about $R^2/2$, half a basis point for a 1% day.

Whatever the choice, the type of return has to be stated. A "return of 10%" is ambiguous: as a log return it corresponds to a simple return of $e^{0.1} - 1$, about 10.5%. On a single day the ambiguity costs little; over a decade, or for a volatile asset such as Bitcoin, it can change conclusions.

## Recap: Simple and Log Returns {h=68cd9043}

The simple return is $R_t = P_t/P_{t-1} - 1$ and the log return $r_t = \ln(P_t/P_{t-1}) = \ln(1 + R_t)$. For every non-zero change $r < R$, and the difference is approximately $R^2/2$, negligible for daily data and large for crashes, as on 12 March 2020, when Bitcoin fell by 37.2% in simple terms and by 46.5% in log terms.

Log returns add up over time; simple returns add up across assets. Gains and losses of the same size do not cancel: +50% followed by −50% leaves 75% of the initial wealth.

## @end

The two returns carry the same information but behave differently under aggregation. Log returns @eq:log add over time and are the natural input of statistical models; simple returns @eq:simple average exactly across assets and are what investors see. The gap, about $R^2/2$, is 0.5 basis points on a 1% day and 9.3 percentage points on Bitcoin's worst day. The [next section](#sec-4) uses both properties to build returns over several periods and returns of portfolios.

# Multi-Period and Portfolio Returns

Investors hold assets for months or years and usually hold several of them. This section builds the return over $k$ periods from daily returns, links cumulative returns to wealth, and computes the return of a portfolio. The two aggregation rules of the previous section, sums over time for log returns and weighted averages across assets for simple returns, are applied to real data: four markets over 2015–2026 and a portfolio of two BVB stocks.

## Returns over k Periods {h=c193522e}

Over $k$ periods ending at $t$, from $P_{t-k}$ to $P_t$, the gross simple returns multiply and the log returns add:

$$ 1 + R_t(k) = \frac{P_t}{P_{t-k}} = (1 + R_t)(1 + R_{t-1}) \cdots (1 + R_{t-k+1}), \qquad r_t(k) = \ln\frac{P_t}{P_{t-k}} = r_t + r_{t-1} + \dots + r_{t-k+1} . $$ {#eq:kper}

Both identities in @eq:kper follow from writing $P_t/P_{t-k}$ as a product of $k$ one-period ratios; taking logarithms turns the product into a sum. Here $R_t(k)$ and $r_t(k)$ denote the simple and the log return over the $k$ periods.

The additivity of log returns has an important consequence: the mean and the variance of the $k$-day log return follow from those of the daily returns. The mean of a sum is the sum of the means, so the $k$-day mean is $k$ times the daily mean; if the daily returns are uncorrelated, the variance of the sum is also $k$ times the daily variance. This is the basis of the annualisation rules of the [next section but one](#sec-5).

Weekly, monthly or annual returns are computed from the prices at the end of each period, not by averaging the daily returns within it. The average of 21 daily returns is a daily figure, not a monthly return; the monthly log return is the sum of the daily ones, or equivalently the log of the ratio of the two month-end prices.

## Cumulative Return and Wealth {h=08c9f520}

Starting from wealth $W_0$, the wealth after $T$ periods is

$$ W_T = W_0 \prod_{t=1}^{T} (1 + R_t) = W_0 \exp\Big(\sum_{t=1}^{T} r_t\Big), $$ {#eq:wealth}

where $\prod_{t=1}^{T}$ is the product of the $T$ gross returns and $\exp(x) = e^x$ inverts the logarithm. The two forms in @eq:wealth are the two identities of @eq:kper applied over the whole sample.

The **cumulative simple return** $\text{CSR}_T = W_T/W_0 - 1$ is the investor's gain in percent. The **cumulative log return** $\text{CLR}_T = \ln(W_T/W_0) = \sum_t r_t$ is the sum of the log returns, and the two are linked by $\text{CSR}_T = e^{\text{CLR}_T} - 1$. The cumulative log return is the natural scale for long horizons: it grows linearly when the growth rate is constant, while the cumulative simple return grows exponentially.

This explains why wealth charts use a logarithmic vertical axis. On a log scale, equal vertical distances correspond to equal percentage changes: a doubling from 100 to 200 and one from 1,000 to 2,000 have the same height. A constant growth rate appears as a straight line, and the slope of the line is the log growth rate.

## Growth of 100 Invested, 2015–2026 {h=a06b9f29}

The chart follows 100 units invested on 5 January 2015, the first day of 2015 on which all four series have prices, on a logarithmic vertical axis. The CAGR tables later in the chapter start each series on its own first trading day of 2015, so their growth rates differ slightly from those implied by this chart: 11.9% against 12.0% for the S&P 500, 60.6% against 62.5% for Bitcoin. At the end of the sample the investment is worth 379 in the S&P 500, 267 in the DAX, 470 in the BET and 29,475 in Bitcoin. The DAX is a total return index while the S&P 500 and the BET are price indices, so the first three lines do not compare like with like, as noted for the [BET and BET-TR](#s15).

For Bitcoin, the cumulative simple return is 29,375% and the cumulative log return is 5.69; for the BET, the two figures are 370% and 1.55. The pairs show the different scales of the two measures: a cumulative simple return of tens of thousands of percent corresponds to a cumulative log return of less than six, and the relation $\text{CSR}_T = e^{\text{CLR}_T} - 1$ links them in both cases.

The logarithmic axis is what makes the chart readable. On a linear axis Bitcoin's line would rise to almost 30,000 and all the other series would appear as flat lines near the bottom. On the log axis the common shock of March 2020 is visible in all four series, as are Bitcoin's rise in 2017 and its fall in 2018.

## Portfolio Returns {h=61617832}

A portfolio holds $n$ assets with weights $w_1, \dots, w_n$, $\sum_i w_i = 1$, fixed at the start of the period, where $w_i$ is the share of wealth invested in asset $i$. Let $R_i$ and $r_i$ be the simple and the log return of asset $i$ over the period. Since wealth is additive across assets, the value of the portfolio at the end of the period is $\sum_i w_i (1 + R_i)$ times its initial value, and the portfolio return is exactly the weighted average of the simple returns:

$$ R_p = \sum_i w_i R_i, \qquad r_p = \ln\Big(\sum_i w_i e^{r_i}\Big) \neq \sum_i w_i r_i . $$ {#eq:port}

The log return of the portfolio, in @eq:port, is the logarithm of a weighted sum and not a weighted sum of logarithms. For example, with 60% in asset 1, with $R_1 = 10\%$, and 40% in asset 2, with $R_2 = -5\%$, the portfolio return is $R_p = 0.6 \times 10\% + 0.4 \times (-5\%) = 4.0\%$, and the exact portfolio log return is $r_p = \ln(1 + R_p) = 3.92\%$. The weighted log returns give $0.6 \times 9.53\% + 0.4 \times (-5.13\%) = 3.67\%$, which is too small. For weights between zero and one the weighted log return is always too small, by the convexity of the exponential (Jensen's inequality).

The weights do not stay fixed. When one asset rises more than the other its share of wealth grows, so a portfolio with fixed weights has to be **rebalanced**, by selling part of the winners and buying the losers. The formula $R_p = \sum_i w_i R_i$ uses the weights at the start of each period, so a portfolio that is rebalanced daily applies it day by day.

## Real Data: a TLV–SNP Portfolio {h=0a724afc}

The portfolio holds 50% Banca Transilvania (TLV) and 50% OMV Petrom (SNP), rebalanced daily, over 2015–2026. Each day it is possible to compare the exact portfolio log return, $\ln(1 + R_p)$, with the weighted sum of the two log returns. The gap is small: its mean in absolute value is 0.38 basis points, and its largest value is 13.9 basis points, on 31 May 2016.

Over the whole period the small daily gaps accumulate, and always in the same direction. The exact cumulative log return of the portfolio is 2.56, so 1 leu becomes 13.0 lei. The weighted log returns give a cumulative log return of 2.47, which corresponds to only 11.8 lei. The error of more than one leu out of thirteen arises entirely from summing an approximation that is very accurate on a single day.

The growth rate of the portfolio is 24.5% per year, against 27.3% for TLV and 19.7% for SNP. The rule is therefore to build portfolios with simple returns, using @eq:port, and to convert to log returns afterwards if needed, with $r_p = \ln(1 + R_p)$.

## Recap: Multi-Period and Portfolio Returns {h=6bd0c4db}

Over time, gross simple returns multiply and log returns add. Wealth follows $W_T = W_0 e^{\text{CLR}_T}$, and the cumulative simple return is $\text{CSR}_T = e^{\text{CLR}_T} - 1$, as for Bitcoin's 29,375%, which corresponds to a cumulative log return of 5.69.

Across assets, $R_p = \sum_i w_i R_i$ is exact, while the weighted sum of log returns is only an approximation. Its daily error is tiny, 0.38 basis points on average for the TLV–SNP portfolio, but over a decade it becomes visible: 11.8 lei instead of 13.0.

## @end

The section has fixed the two aggregation rules of the course: log returns for sums over time @eq:kper, simple returns for averages across assets @eq:port. Wealth is the exponential of the cumulative log return @eq:wealth, which is why long histories are plotted on a logarithmic axis. Investors, however, compare assets per year, not per day or per decade. The [next section](#sec-5) converts daily statistics into annual ones and measures how precisely the annual mean can be known.

# Annualisation

Statistics computed from daily data are daily figures, while returns and risks are usually quoted per year. This section gives the rules for annualising the mean and the volatility, defines the compound annual growth rate (CAGR) and shows that, unlike volatility, the mean return is estimated very imprecisely, however frequent the data.

## Annualising the Mean and the Volatility {h=ef18305e}

Let $A$ be the number of observations per year: about 252 for an exchange, 365 for crypto, 52 for weekly data and 12 for monthly data. Let $\mu$ and $\sigma$ be the mean and the standard deviation of the returns at the frequency of the data, for example daily. Since log returns add up over a year, the annual mean is the sum of $A$ daily means, and the same rule is used for the mean of simple returns, as an arithmetic annualisation. The volatility, the standard deviation of the returns, scales with the square root of $A$:

$$ \mu_{\text{year}} = A\,\mu, \qquad \sigma_{\text{year}} = \sqrt{A}\,\sigma . $$ {#eq:annual}

The second rule in @eq:annual is the **square-root-of-time rule**. It holds when the returns are uncorrelated and have constant variance: then $\Var(r_1 + \dots + r_A) = A\,\sigma^2$, because all the covariances vanish, and the square root gives $\sqrt{A}\,\sigma$. With positive autocorrelation the annual variance is larger than $A\sigma^2$, and with negative autocorrelation smaller; this is the idea behind the variance ratio tests of [Chapter 7](ch7.html).

For the S&P 500, the daily standard deviation is 1.12%, so the annual volatility is $1.12\% \times 15.87 = 17.7\%$, where $15.87 = \sqrt{252}$. For Bitcoin, which trades every day, the factor is $\sqrt{365} = 19.10$, which gives 66.7%. The exchange factor $\sqrt{252}$ would give only 55.4% and understate Bitcoin's risk by more than 11 percentage points. This is why the frequency rule of the [data section](#s8) matters.

## The Compound Annual Growth Rate {h=216c343a}

The **CAGR** (compound annual growth rate) is the constant annual rate that turns the first price $P_0$ into the last price $P_T$ in $Y$ years:

$$ \text{CAGR} = \big(P_T / P_0\big)^{1/Y} - 1 , $$ {#eq:cagr}

with $Y$ measured in calendar years. Taking logarithms of @eq:cagr gives $\ln(1 + \text{CAGR}) = \ln(P_T/P_0)/Y$, the cumulative log return per year. Hence $\text{CAGR} = \exp(\text{annual mean of log returns}) - 1$.

For the S&P 500 over 2015–2026, the annual mean of the log returns is 11.2%, and $e^{11.2\%} - 1 = 11.9\%$, the same value as the CAGR, 11.9%. The CAGR is also called the geometric mean return, because $1 + \text{CAGR}$ is the geometric mean of the annual gross returns.

The CAGR uses only the first and the last price. It says nothing about the path between them: two investments with the same CAGR can have had very different volatilities and very different falls along the way. The indicators of the [performance section](#sec-7) measure these differences.

## Precision of the Estimated Mean {h=e9c0148f}

The standard error (SE) of the estimated annual mean $\hat\mu_{\text{year}}$, computed from $Y$ years of data, is

$$ \text{SE}(\hat\mu_{\text{year}}) = \frac{\sigma_{\text{year}}}{\sqrt{Y}} , $$ {#eq:se-mean}

where the hat denotes a value estimated from the sample. The formula depends on the number of **years**, not on the number of observations. The 95% confidence interval is $\hat\mu_{\text{year}} \pm 1.96\,\text{SE}$, where 1.96 is the 97.5% quantile of the standard Normal distribution.

For the S&P 500, with 11.7 years of data, the annual mean of the log returns is 11.2% and its SE is 5.2%. The 95% confidence interval is [1.1%; 21.4%]: the data are compatible with an expected return that is barely positive and with one that is almost twice the estimate. For Bitcoin the estimate is 47.4% with an SE of 19.5%, and the interval, $47.4\% \pm 1.96 \times 19.5\%$, is [9.2%; 85.6%], very wide.

Volatility, in contrast, is estimated precisely from daily data, because every daily observation carries information about the size of the moves. The asymmetry between the two estimates, an imprecise mean and a precise volatility, recurs throughout the course, and it is the reason why ratios that divide a mean by a volatility, such as the [Sharpe ratio](#s52), inherit the imprecision of the mean.

## Question for the Room: More Data, Better Mean? {h=5a67a27d}

Suppose that 10 years of daily S&P 500 data are replaced by 10 years of 5-minute data: 78 bars in a trading day of 6.5 hours, so 78 times more observations. What do you think: does the standard error of the annual mean return become smaller?

It does not. The mean log return over 10 years is determined by the first and the last price alone, $\hat\mu = \ln(P_T/P_0)/Y$, because the sum of the log returns telescopes: every intermediate price appears once with a plus sign and once with a minus sign. Sampling more often adds observations between the same two endpoints but no information about the mean. Only more **years** reduce its standard error, in proportion to $1/\sqrt{Y}$.

The estimate of volatility, on the other hand, does improve with frequency, since the squared returns of each 5-minute interval carry information about the variance. This is the principle of realised volatility in [Chapter 8](ch8.html), up to the limit at which microstructure noise starts to dominate.

## Recap: Annualisation {h=37666799}

The mean is annualised by multiplying by $A$, the volatility by multiplying by $\sqrt{A}$, with $A$ the actual number of observations per year of the series. The square-root rule assumes uncorrelated returns with constant variance.

The CAGR, $(P_T/P_0)^{1/Y} - 1$, equals the exponential of the annual mean of log returns minus one. Mean returns are imprecise: their SE is $\sigma_{\text{year}}/\sqrt{Y}$, 5.2% per year for the S&P 500 over 11.7 years.

## @end

Annualisation follows from the aggregation rules of the previous section: means add over time and, without autocorrelation, so do variances, which gives @eq:annual. The CAGR @eq:cagr summarises growth by the first and last price, and the precision of the mean @eq:se-mean depends only on the length of the sample in years. With the S&P 500's 95% interval for the annual mean running from 1.1% to 21.4%, the mean is the least reliable statistic of a return series. The [next section](#sec-6) describes the other features of the distribution of returns, which the data estimate much better.

# Descriptive Statistics of Returns

A return series is described first by its location and spread, then by the shape of its distribution: asymmetry, the weight of the tails and the extremes. This section defines these statistics, computes them for the indices, Bitcoin and five BVB stocks, compares the histograms with the Normal distribution and shows that volatility changes over time.

## Location and Spread {h=64509964}

Consider a sample $r_1, \dots, r_n$ of daily log returns, in percent. The **mean** $\bar r = \frac1n \sum_{t=1}^{n} r_t$ is the average daily return. The **variance** $\hat\sigma^2 = \frac{1}{n-1} \sum_{t=1}^{n} (r_t - \bar r)^2$ uses the divisor $n - 1$, which makes it unbiased, and the **standard deviation** is $\hat\sigma = \sqrt{\hat\sigma^2}$. In finance, the standard deviation of returns is called **volatility**.

The **quantile** $q_\alpha$ is the value below which a fraction $\alpha$ of the returns lies. The median is $q_{0.5}$. The 1% quantile $q_{0.01}$ is the basis of the value at risk (VaR): VaR 1% is $\text{VaR}_{0.01} = -q_{0.01}$, the loss exceeded on 1% of days, written as a positive number. VaR and its backtesting are the subject of [Chapter 10](ch10.html).

The **minimum** and the **maximum** are the worst and the best day of the sample. They should always be reported with their dates, because a date identifies the event behind an extreme value and is the quickest way to tell a genuine crash from a data error.

## Shape: Skewness and Kurtosis {h=0898729c}

The **skewness** measures the asymmetry of the distribution:

$$ \hat S = \frac{\frac1n \sum_t (r_t - \bar r)^3}{\hat\sigma^3}, \qquad \hat K = \frac{\frac1n \sum_t (r_t - \bar r)^4}{\hat\sigma^4} . $$ {#eq:moments}

The third power in @eq:moments keeps the sign of each deviation, so large losses push $\hat S$ below zero and large gains above zero. A symmetric distribution has $S = 0$; $S < 0$ indicates a longer left tail, with large losses more frequent than large gains of the same size.

The **kurtosis** $\hat K$ uses the fourth power, so the extreme days dominate it. The Normal distribution has $K = 3$, and the **excess kurtosis** $\hat K - 3$ is therefore zero for the Normal distribution. A positive excess kurtosis indicates **heavy tails**: extreme days more frequent than under the Normal distribution with the same mean and standard deviation.

Both statistics are sensitive to isolated extreme days: one crash can change the skewness from positive to negative and double the kurtosis. They should therefore be reported together with the extremes. Heavy tails and negative skewness are **stylised facts** of returns, regularities found across markets and periods [@Cont2001]; the formal tests are in [Chapter 2](ch2.html).

## Daily Log Returns: Indices and Bitcoin, 2015–2026 {h=d8eb0f5e}

The table describes the daily log returns of four indices and Bitcoin, each on its own calendar; all values except the skewness and the excess kurtosis are in percent. The S&P 500 has a mean of 0.045% per day and a standard deviation of 1.12%, the DAX 0.032% and 1.20%, the BET 0.053% and 0.98%, the BET-TR 0.080% and 0.98%. Bitcoin's mean is 0.130% and its standard deviation 3.49%, about three times that of the stock indices. In every row the daily mean is a small fraction of the standard deviation, which is why the mean is so hard to estimate.

All five series have negative skewness, from −0.56 for the DAX to −1.43 for the BET-TR, and large excess kurtosis, from 10.0 for the DAX to 18.1 for the BET-TR. The BET and the BET-TR have the same standard deviation and almost the same shape, since dividends shift the mean but hardly change the daily fluctuations.

The extremes have dates. The worst day was 16 March 2020 for the S&P 500, at −12.8%, and 12 March 2020 for the DAX, at −13.1%, and for Bitcoin, at −46.5%: all in the market crash at the start of the COVID-19 pandemic. For the BET and the BET-TR the worst day was 19 December 2018, at −11.9%, a Romanian event explained in the next frame. The best days, from 6.7% for the BET-TR to 22.5% for Bitcoin, are smaller in absolute value than the worst ones, in line with the negative skewness.

## Daily Log Returns: Five BVB Stocks, 2015–2026 {h=28ee24be}

The second table describes five BVB stocks: Banca Transilvania (TLV), OMV Petrom (SNP), BRD, Nuclearelectrica (SNN) and Transgaz (TGN). Their daily standard deviations lie between 1.53% for SNN and 1.69% for TLV, all well above the 0.98% of the BET index. This is the effect of diversification: in an index, the movements specific to individual companies partly offset each other.

Banca Transilvania and BRD had their worst day on 19 December 2018, with −22.2% and −18.2%, after the announcement of a tax on bank assets, adopted by the Government Emergency Ordinance (OUG) 114/2018. Transgaz also had its worst day then, at −10.1%, while SNP and SNN had theirs on 9 March 2020, at −15.0% and −9.4%. A single policy announcement thus produced the worst day of the whole period for the Romanian market.

The shape of the distribution differs much between stocks. TLV has an excess kurtosis of 25.2 and SNN only 5.2; SNN and TGN even have slightly positive skewness, 0.29 and 0.01. These differences come largely from a few days, which is why the extremes and their dates have to be read with the moments. For TLV one of these days is a data error: the EODHD adjusted close of 30 May 2016 is wrong, which produces a return of −15.0% on that day and of +20.5%, the maximum in the table, on 31 May. The table keeps both days; without them the excess kurtosis of TLV is 18.1. Crashes on the BVB have their own statistical description, studied by @PeleMazurencuMarinescu2012.

## Histograms vs the Normal Distribution {h=a0474a8d}

The chart compares the histograms of the daily log returns of the S&P 500 (left) and Bitcoin (right) with the density of the Normal distribution with the same mean and standard deviation (black line). The horizontal axis is cut at the 0.1% and 99.9% quantiles of each series; the few days beyond them are counted in the outermost bars. Both histograms have the same characteristic shape: a centre that is much higher and narrower than the Normal density, shoulders that are lower, and tails that extend further.

For the S&P 500, 79.6% of days lie within one standard deviation of the mean, against 68.3% under the Normal distribution: there are more quiet days than the Normal distribution predicts. Beyond three standard deviations lie 1.39% of days, against 0.27% under the Normal distribution, that is, 5.2 times as many.

The two effects are linked. With the standard deviation fixed, a distribution that places more days in the tails must place more days near the centre as well, and fewer in between. A Normal model fitted to these data therefore underestimates both the frequency of quiet days and, more importantly for risk, the frequency of extreme days. Measures of risk that assume normality, such as a Normal VaR, inherit this error.

## Time-Varying Volatility {h=c678a694}

The chart shows the annualised volatility of the S&P 500, the BET and Bitcoin, computed on rolling windows of one year. The volatility of the S&P 500 ranges between 7% and 35%, that of Bitcoin between 41% and 106%. The BET follows a pattern similar to the S&P 500, with a jump in 2020.

The shape of the lines comes from the rolling window. When an extreme day enters the window, the estimate jumps up and stays high for exactly one year, until the day leaves the window; this produces the plateaus visible, for example, for the S&P 500 from March 2020 to March 2021. The rolling estimate thus reacts late to calm periods and smooths turbulent ones, but it shows clearly that volatility is not constant: calm and turbulent periods alternate and each lasts for months.

A single volatility for the whole sample hides these periods. The tendency of large moves to follow large moves, and small moves to follow small ones, is volatility clustering. Better estimators of volatility are the subject of [Chapter 8](ch8.html), and the GARCH models (generalised autoregressive conditional heteroskedasticity), which make the variance depend on the recent past, of [Chapter 9](ch9.html).

## Recap: Descriptive Statistics {h=e05faa63}

A return series should be reported with its mean, standard deviation, skewness, excess kurtosis, quantiles and extremes with their dates. Daily returns typically have a mean close to zero, a volatility of 1.0–1.7% per day for the indices and the BVB stocks of the chapter and 3.5% for Bitcoin, heavy tails and often negative skewness.

Extreme days are several times more frequent than under the Normal distribution: 5.2 times beyond three standard deviations for the S&P 500. Volatility is not constant over time: on one-year windows, that of the S&P 500 ranged from 7% to 35%.

## @end

The descriptive statistics of daily returns reveal three regularities, the same on the indices, the BVB stocks and Bitcoin: means that are small relative to the standard deviation, heavy tails with negative skewness, and volatility that changes over time. Each has a consequence for the indicators of the [next section](#sec-7): the small mean makes return-based indicators imprecise, heavy tails make their standard errors larger than the formulas assume, and changing volatility makes the deepest falls concentrate in a few turbulent periods.

# Performance Indicators

Investors compare investments not only by how much they grew but also by how much risk they carried and how deep their falls were. This section defines the standard risk-adjusted indicators, the Sharpe, Sortino and Calmar ratios, the drawdown and volatility drag, applies them to the assets of the chapter and shows that each of them is an estimate with a sizeable error.

## Limits of a Single Performance Indicator {h=e12abc3c}

Return alone rewards risk-taking. A leveraged position, financed partly with borrowed money, multiplies both gains and losses, so it can show a high return for years, until a fall wipes it out. An indicator that looks only at return cannot distinguish a skilled manager from one who simply took more risk.

An investor therefore asks three questions. How much did the investment grow? The answer is the CAGR. How large were the fluctuations? The answers are the volatility and the downside deviation. How deep was the largest fall? The answer is the maximum drawdown (MDD). Risk-adjusted indicators combine these answers by dividing a return by a measure of risk: the Sharpe ratio divides by volatility, the Sortino ratio by downside deviation, the Calmar ratio by the maximum drawdown.

All these indicators are estimates from a single sample, and each has an estimation error. A ranking of assets by an indicator is informative only if the differences between them are large relative to these errors. This point recurs in every frame of the section.

## The Sharpe Ratio {h=51f85276}

The **Sharpe ratio** [@Sharpe1966; @Sharpe1994] is the excess return per unit of total risk:

$$ \text{SR} = \frac{\mu - r_f}{\sigma}, \qquad \text{SR}_{\text{year}} = \sqrt{A}\,\text{SR}_{\text{day}} , $$ {#eq:sharpe}

where $\mu$ is the mean return, $r_f$ the risk-free rate and $\sigma$ the volatility. The annual ratio follows from @eq:annual: the numerator is multiplied by $A$ and the denominator by $\sqrt{A}$, so the ratio is multiplied by $\sqrt{A}$.

In this chapter the Sharpe ratio uses the mean of daily simple returns and $r_f = 0$. The reason is practical: the assets are quoted in lei, euro and dollars, each currency with its own risk-free rate, and setting $r_f = 0$ for all keeps the comparison simple. The ratio is then a mean return per unit of volatility rather than a strict excess return. For the S&P 500 the daily Sharpe ratio is 0.046, and the annual one is 0.72.

The Sharpe ratio does not depend on leverage. Borrowing at the risk-free rate to double a position doubles both the excess return and the volatility, so the ratio stays the same. Two investments with the same Sharpe ratio thus offer the same return per unit of risk, and an investor can reach any desired level of risk with either. William F. Sharpe, shown on the slide, received the Nobel Prize in Economics in 1990.

## Estimation Error of the Sharpe Ratio {h=6d29e0f9}

For returns that are i.i.d. (independent and identically distributed), @Lo2002 gives the standard error of the estimated Sharpe ratio $\widehat{\text{SR}}$:

$$ \text{SE}(\widehat{\text{SR}}) \approx \sqrt{\frac{1 + \text{SR}^2/2}{n}} , $$ {#eq:se-sr}

per period, where $n$ is the number of returns; the annual standard error is obtained by multiplying by $\sqrt{A}$, the number of returns per year. For daily data the per-period ratio is small, so the term $\text{SR}^2/2$ hardly matters, and the annual standard error is about $\sqrt{A/n} = 1/\sqrt{Y}$: around 0.3 with 12 years of data, whatever the asset.

For the S&P 500 over 2015–2026, $\text{SR} = 0.72$ with a standard error of 0.29, and the 95% confidence interval is [0.15; 1.30]. The interval excludes zero, but it is compatible both with a modest and with an exceptionally good risk-return trade-off. Like the mean, the Sharpe ratio is known only roughly after more than a decade of data.

The real error is larger than @eq:se-sr suggests, because heavy tails and volatility clustering violate the i.i.d. assumption. A second source of error is selection: choosing the best of many strategies overstates the Sharpe ratio of the strategy chosen, and @BaileyLopezDePrado2014 propose a correction for this, the deflated Sharpe ratio.

## Sharpe Ratios with 95% Intervals, 2015–2026 {h=d2ce3fcf}

The chart shows the annualised Sharpe ratios of the ten assets of the chapter, with $r_f = 0$, each with its 95% confidence interval. The highest point estimate is that of the BET-TR, followed by Nuclearelectrica, Banca Transilvania and Bitcoin; the lowest is that of the DAX.

All the intervals overlap, so the data do not support a reliable ranking of these assets by their Sharpe ratio. The interval of the DAX contains zero: over this period its Sharpe ratio is not significantly different from zero. Overlapping intervals are an informal guide; a formal comparison of two Sharpe ratios uses the standard error of their difference, which also depends on the correlation between the two return series.

The widths of the intervals are almost equal, although the assets differ greatly in volatility. This follows from @eq:se-sr: the standard error of an annual Sharpe ratio depends mainly on the length of the sample, about $1/\sqrt{Y}$, and all the series cover the same years.

## The Sortino Ratio {h=47e050ef}

Investors fear losses, not gains, but volatility treats deviations in both directions alike. The **downside deviation** below a threshold $\tau$, here $\tau = 0$, counts only the days below the threshold:

$$ \sigma_D = \sqrt{\frac1n \sum_t \min(R_t - \tau, 0)^2}, \qquad \text{Sortino} = \frac{\mu - \tau}{\sigma_D} . $$ {#eq:sortino}

In @eq:sortino, $\min(R_t - \tau, 0)$ is the shortfall below the threshold on day $t$, zero on the days above it, and $n$ counts all the days of the sample, not only those below the threshold. The **Sortino ratio** [@SortinoVanDerMeer1991] divides the mean return in excess of the threshold by the downside deviation and is annualised like the Sharpe ratio.

For a symmetric distribution centred near the threshold, about half of the squared deviations fall below it, so $\sigma_D \approx \sigma/\sqrt2$ and the Sortino ratio is about $\sqrt2 \approx 1.4$ times the Sharpe ratio. For the S&P 500 over 2015–2026 the Sharpe ratio is 0.72 and the Sortino ratio 1.02, close to this rule. A Sortino ratio well above 1.4 times the Sharpe ratio would indicate that the losses are smaller than the gains; well below, that they are larger.

## Drawdown and Maximum Drawdown {h=b9e66b5b}

The **drawdown** is the fall from the highest price reached so far:

$$ \text{DD}_t = \frac{P_t}{\max_{s \le t} P_s} - 1 \le 0, \qquad \text{MDD} = \min_t \text{DD}_t , $$ {#eq:dd}

where $\max_{s \le t} P_s$ is the highest price over all days $s$ up to and including day $t$. The drawdown is zero at every new high and negative in between. The **maximum drawdown** (MDD) is the largest fall from a peak to the following trough. It should be reported with its dates: the peak, the trough and the **recovery** to the old peak, if it occurred.

The depth of a drawdown matters more than it seems, because of the recovery rule: after a loss $x$, the gain needed to return to the peak is $x/(1 - x)$. A loss of 10% needs 11%, 20% needs 25%, 50% needs 100% and 80% needs 400%. Deep falls are much harder to recover from than their size suggests.

Unlike volatility, the MDD depends on the path of the prices, and it grows with the length of the sample: the longer the history, the more chances for a deep fall [@MagdonIsmailEtAl2004]. Two MDDs are comparable only over periods of the same length.

## Drawdowns over the Full History {h=fecc5dea}

The chart plots the drawdown of the S&P 500 (since 2000), the BET (since 1997) and Bitcoin over their full histories. Each line touches zero at a new high and sinks below zero during a fall; the depth of a dip is the drawdown, its width the time spent below the previous peak.

The S&P 500 had its MDD of −56.8% from 9 October 2007 to 9 March 2009, in the global financial crisis, and recovered its peak on 28 March 2013. The BET had an MDD of −82.5% from 24 July 2007 to 25 February 2009, and recovered only on 16 March 2021. Bitcoin had an MDD of −83.4% from 16 December 2017 to 15 December 2018.

The chart shows how different the three histories are. The S&P 500 spent long stretches near its peak, with two deep falls; the BET spent most of the period from 2008 to 2021 far below its 2007 peak; Bitcoin's line moves constantly between new highs and falls of more than 50%.

## Reading the Drawdown Chart {h=d0ce901f}

The chart is read along two dimensions. The first is depth. After a drawdown of −82.5%, the BET needed a gain of +473% to recover; after −83.4%, Bitcoin needed +502%. These figures follow from the recovery rule $x/(1-x)$ and show why a drawdown of more than 80% is so damaging.

The second is duration. The BET needed 13.6 years to regain its 2007 peak. The share of days spent more than 20% below the previous peak is 27% for the S&P 500, 56% for the BET and 66% for Bitcoin. An investor in the BET or in Bitcoin spent most of the time with a loss of more than a fifth relative to the best moment.

Volatility and the Sharpe ratio ignore the order of returns: shuffling the days changes neither. The drawdown takes the order into account, because a run of losses produces a deeper fall than the same losses spread among gains. Over the comparison period 2015–2026 the MDD of each stock index occurred in February–March 2020, while Bitcoin and several BVB stocks had theirs at other dates.

## The Calmar Ratio {h=b903ecd2}

The **Calmar ratio** measures growth per unit of the largest fall:

$$ \text{Calmar} = \frac{\text{CAGR}}{|\text{MDD}|} . $$ {#eq:calmar}

It is popular among fund managers because the MDD is a loss the investor actually bears, while volatility is an abstract measure of dispersion. For the S&P 500 over 2015–2026, the CAGR is 11.9% and the MDD −33.9%, so the Calmar ratio is $11.9/33.9 = 0.35$.

The denominator of @eq:calmar is determined by a single event, the deepest fall of the sample. One more crash, or one less, changes the ratio much more than it would change the volatility, so the Calmar ratio is even more unstable than the Sharpe ratio. Since the MDD also grows with the length of the sample, Calmar ratios should only be compared over the same period.

## Volatility Drag {h=c26de569}

For a log return with mean $m$ and variance $\sigma^2$, the simple return has mean approximately $m + \sigma^2/2$; for normally distributed log returns this is exact, since $\E[e^{r}] = e^{m + \sigma^2/2}$. Hence the growth rate of wealth, the mean log return, lies below the mean simple return:

$$ \mu_{\text{log}} \approx \mu_{\text{arith}} - \frac{\sigma^2}{2} , $$ {#eq:drag}

where $\mu_{\text{arith}}$ is the mean of the simple returns and $\mu_{\text{log}}$ the mean of the log returns. The difference $\sigma^2/2$ is **volatility drag** [@BoothFama1992]. It comes from the concavity of the logarithm, through Jensen's inequality, and it is the same effect as in the example of +10% followed by −10%.

Consider two assets with the same arithmetic mean, 10% per year. With $\sigma = 5\%$, the drag is 0.12%, the growth rate 9.88% per year, and after 10 years 1 leu becomes 2.68. With $\sigma = 30\%$, the drag is 4.50%, the growth rate 5.50% per year, and after 10 years 1 leu becomes only 1.73. The same average return produces very different wealth.

The drag grows with the square of volatility, so doubling the volatility quadruples the drag. For low-volatility assets it is negligible; for volatile ones it becomes the main difference between the average return and the actual growth of wealth.

## Volatility Drag on Real Data {h=b523b9a9}

The chart tests @eq:drag on the assets of the chapter over 2015–2026. Each point is an asset. The vertical axis shows the annual arithmetic mean minus the annual log mean, the drag observed in the data; the horizontal axis shows $\sigma^2/2$, the drag predicted by the formula. Both axes are logarithmic, and the dashed line marks equality.

All the points lie on the line, from the indices, with a drag of about 1.2 to 1.8 percentage points per year, through the BVB stocks, between about 2.6 and 3.3, up to Bitcoin. For Bitcoin the arithmetic mean is 69.5% per year and the log mean 47.4%, so the observed drag is 22.1 percentage points, while $\sigma^2/2 = 21.9\%$.

The close fit shows that the gap between the two means is not a statistical accident but a consequence of volatility. For Bitcoin the drag removes almost a third of the arithmetic mean.

## Question for the Room: Which Bitcoin Return? {h=7e14f7f8}

For Bitcoin over 2015–2026, the arithmetic mean of the simple returns is 69.5% per year and the CAGR is 60.6%. What do you think: which of the two figures describes the growth of an investor's wealth?

The answer is the CAGR: it is the rate that actually turned the first price into the last. The arithmetic mean is the expected return over a single year, not the growth rate over many years.

The gap between the two is volatility drag. In log terms the drag is 22.1 percentage points: from the arithmetic mean of 69.5% to a mean log return of 47.4%. Expressed as an annual compound rate, a log growth of 47.4% per year is a CAGR of $e^{0.474} - 1 = 60.6\%$. For assets with low volatility the two figures almost coincide; for Bitcoin, quoting the arithmetic mean as "the return" overstates growth by about nine percentage points per year.

## Performance Indicators: Indices and Bitcoin, 2015–2026 {h=705e35b7}

The table brings together the indicators of the section for the four indices and Bitcoin. All are annualised with the actual frequency of each series; the volatility (Vol.) is that of daily simple returns, and the Sharpe and Sortino ratios use $r_f = 0$. The drag column is the observed difference between the arithmetic and the log mean, in percentage points.

Bitcoin has by far the highest CAGR, 60.6%, but also the highest volatility, 66.2%, and the deepest drawdown, −83.4%; its drag of 22.13 percentage points is more than ten times that of any index. Its Sharpe ratio, 1.05, is in the range of the others, and its Calmar ratio, 0.73, is close to that of the BET-TR, 0.71. Among the indices, the DAX has the lowest CAGR, 8.5%, and the lowest Sharpe ratio, 0.52, while the S&P 500 lies in between, with 11.9% and 0.72.

The comparison of the BET-TR with the BET isolates the effect of dividends. The volatility, 15.4%, and the MDD, −31.1%, are the same, because dividends hardly change the daily fluctuations. Every indicator built on returns is higher: the CAGR is 22.1% against 14.1%, the Sharpe ratio 1.38 against 0.93, the Calmar ratio 0.71 against 0.45.

## Performance Indicators: BVB Stocks, 2015–2026 {h=27d1734a}

The second table gives the same indicators for the five BVB stocks, computed from adjusted prices, so dividends are included. Nuclearelectrica has the highest CAGR, 30.7%, the highest Sharpe ratio, 1.27, the highest Sortino ratio, 2.00, and the highest Calmar ratio, 0.83. Banca Transilvania follows with a CAGR of 27.3% and a Sharpe ratio of 1.08. BRD has the lowest CAGR, 18.9%, and the lowest Sharpe ratio, 0.83.

Individual stocks have higher volatility than the BET index, between 23.0% and 25.5% against 15.4%, and deeper drawdowns, from −35.6% for BRD to −45.4% for OMV Petrom against −31.1%. This is the diversification effect seen in the descriptive statistics, now in the risk indicators.

These differences must be read against the estimation error. Each Sharpe ratio has a standard error of about 0.29, so a difference such as that between Transgaz, 0.98, and OMV Petrom, 0.89, is far from significant. Even the gap between the best and the worst stock is less than twice the standard error of a single ratio.

## Risk and Return, 2015–2026 {h=63d3b27e}

The chart places each asset by its annualised volatility (horizontal axis) and its CAGR (vertical axis). Bitcoin lies far to the upper right. The BVB stocks form a group, with volatilities between 23.0% and 25.5% and CAGRs between 18.9% and 30.7%; the indices lie to the left, with lower volatility.

The ranking depends on the indicator. By CAGR the best asset is Bitcoin; by the Sharpe ratio, the BET-TR; by the Sortino and the Calmar ratios, Nuclearelectrica (SNN). The lowest volatility belongs to the BET-TR, 15.4% as for the BET at the precision shown in the tables. The smallest MDD in absolute value also belongs to the BET-TR, −31.115% against −31.124% for the BET: practically a tie. Each indicator answers a different question, so each produces its own ranking.

The practical rule is to state the indicator whenever an asset is called "the best". A statement such as "Bitcoin was the best investment of 2015–2026" is true for growth and false for risk-adjusted return measured by the Sharpe ratio.

## Recap: Performance Indicators {h=57526b35}

The CAGR measures growth, volatility measures risk, and the Sharpe and Sortino ratios measure return per unit of total or downside risk. The MDD and the Calmar ratio describe the largest fall, the loss the investor actually bears.

Volatility drag, $\mu_{\text{log}} \approx \mu_{\text{arith}} - \sigma^2/2$, places growth below the arithmetic mean, by 22.1 percentage points per year for Bitcoin. Each indicator is an estimate, and differences between assets must be compared with their standard errors, about 0.29 for a Sharpe ratio estimated over 2015–2026.

## @end

The indicators of the section answer different questions, and on the same data they rank the assets differently: Bitcoin leads by growth, the BET-TR by the Sharpe ratio, Nuclearelectrica by the Sortino and Calmar ratios. The Sharpe ratio @eq:sharpe has a standard error @eq:se-sr of about 0.29 over 2015–2026, so none of these rankings is statistically secure, and the Calmar ratio @eq:calmar, driven by a single fall, is even less stable. Volatility drag @eq:drag links the arithmetic mean to the growth of wealth. The [next section](#sec-8) turns the question of estimation error into an open research question.

# AI for Scientific Discovery

The chapter ends with an open question that its methods can address, and with the role that artificial intelligence (AI) tools can play in studying it. The question is whether Sharpe ratios persist over time. The section describes what AI can contribute, which checks remain the researcher's responsibility, and a project that applies the chapter's indicators to BVB stocks.

## An Open Research Question {h=487fa523}

**Do Sharpe ratios persist over time?** If a BVB stock had a high Sharpe ratio in 2015–2020, did it also have a high one in 2021–2026? The answer matters in practice. If past Sharpe ratios do not persist, ranking funds or stocks by their past Sharpe ratio, a common practice, is of little use to an investor choosing for the future.

The question is open for the BVB for statistical reasons. The samples are short, the number of liquid stocks is small, and a Sharpe ratio estimated on half of the chapter's sample has a standard error of about 0.4, by @eq:se-sr. With errors of this size, true persistence and pure noise are hard to tell apart.

Related results warn against optimism: many return regularities published in the literature do not hold up on new data [@HarveyLiuZhu2016]. AI tools can speed up a study of this kind, from the search of the literature to the code, but they do not replace its verification [@WangEtAl2023].

## Possible Uses of AI Tools {h=51fbc7fe}

An AI assistant can contribute at four stages of such a study. For the **literature**, it can find papers on the persistence of performance and summarise their methods. For the **code**, it can write a first version of a Python function that computes the Sharpe ratio on rolling windows of the course data.

For **robustness**, it can propose other windows, other indicators, such as the Sortino and Calmar ratios, and other markets. For the **explanation**, it can turn a table of results into a first draft of the interpretation, which the researcher then checks against the numbers.

The prompt on the slide shows the level of precision a useful request needs: it specifies the input (daily simple returns), the indicator (the annualised Sharpe ratio), the windows (non-overlapping, three years) and the method for the standard error (@Lo2002). A vague request produces code whose assumptions the user has to discover afterwards.

## What to Check {h=20ffa8be}

Every output of an AI tool has to be checked, and the chapter provides the checklist. The data: adjusted prices for stocks, the same calendar for all assets, the correct period. The formulas: the annualisation factor, $\sqrt{252}$ or $\sqrt{365}$, simple or log returns, the divisor $n$ or $n - 1$. Each of these choices was shown in the chapter to change the result, for example by 11 percentage points in Bitcoin's annual volatility.

The numbers: recompute two or three values by hand on small examples, or with a second method. The references: every paper cited must exist, and its DOI (digital object identifier) must lead to it; language models can produce plausible references that do not exist.

The selection: if ten windows were tried, the report must say so. A backtest repeated many times, with small changes until the result looks good, leads to overfitting [@BaileyEtAl2014], the same problem as the data snooping of the [first section](#s17).

## Project Idea {h=392190ed}

The project asks whether the Sharpe ratios of BVB stocks persist from one period to the next. The data are the adjusted prices of 10 to 15 BVB stocks over 2010–2026, from the course data.

The method has three steps. First, split the sample into two halves and compute the Sharpe ratio of each stock in each half, with its standard error. Second, measure persistence by the rank correlation between the two halves: a high correlation means that the stocks that ranked high in the first half also ranked high in the second. Third, repeat the analysis with the Sortino and Calmar ratios, to see whether the answer depends on the indicator.

The deliverables are a table, a chart and a paragraph on what the data can and cannot show, given the standard errors. Any use of AI must be declared, together with the AI errors that were found and corrected.

## @end

The persistence of Sharpe ratios is a question that the tools of this chapter can address but not settle on BVB data, because a Sharpe ratio estimated over six years has a standard error of about 0.4. AI tools can shorten the work on literature, code and robustness; the checks of data, formulas, numbers, references and selection remain with the researcher. The [summary](#sec-9) collects the formulas of the chapter.

# Summary

The chapter has built the basic quantities of the course: from a checked price series to returns, their aggregation over time and across assets, their annualisation, their descriptive statistics and the indicators that compare investments. The summary collects the formulas and three questions for self-assessment.

## Key Formulas {h=bc829d39}

The table on the slide collects the formulas of the chapter in the order in which they were introduced. The simple return @eq:simple and the log return @eq:log are the two basic definitions, linked by $r_t = \ln(1 + R_t)$. Over $k$ periods, gross simple returns multiply and log returns add @eq:kper; across assets, the portfolio return is the weighted average of simple returns @eq:port.

Annualisation multiplies the mean by $A$ and the volatility by $\sqrt{A}$ @eq:annual, and the CAGR @eq:cagr gives the constant annual growth rate between the first and the last price. The Sharpe and Sortino ratios divide the mean return by the volatility @eq:sharpe and by the downside deviation @eq:sortino; the MDD @eq:dd and the Calmar ratio @eq:calmar describe the largest fall. Volatility drag @eq:drag links the arithmetic mean to the growth rate.

The last row gives the two standard errors of the chapter: $\sigma_{\text{year}}/\sqrt{Y}$ for the annual mean @eq:se-mean and $\sqrt{(1 + \text{SR}^2/2)/n}$ for the Sharpe ratio @eq:se-sr. Both depend on the length of the sample, and both are large enough to change the reading of most comparisons in the chapter.

## Self-Assessment {h=3a77cd6c}

The first question asks for the annual mean and volatility of a fund whose daily log returns have a mean of 0.05% and a standard deviation of 1%. With 252 trading days, the annual mean is $252 \times 0.05\% = 12.6\%$ and the annual volatility $\sqrt{252} \times 1\% = 15.9\%$, by @eq:annual.

The second question asks why the arithmetic mean of Bitcoin's returns is far above its CAGR. The answer is volatility drag, $\sigma^2/2$, which is large when $\sigma$ is large: for Bitcoin, with an annual volatility of about 66%, the drag is about 22 percentage points per year.

The third question asks what gain an index needs to recover after a fall of 60%. By the recovery rule $x/(1 - x)$, the required gain is $0.6/0.4 = 150\%$. The next chapter, [Chapter 2](ch2.html), studies the classical distributions and the stylised facts of returns, starting from the heavy tails and negative skewness found here.

## @end

From a single price series, the chapter has derived the quantities used in the rest of the course: returns of two kinds, their aggregation, annualised means and volatilities, the moments and extremes of their distribution, and risk-adjusted indicators with their standard errors. Two findings carry forward. The distribution of daily returns has heavy tails and time-varying volatility, the subject of [Chapter 2](ch2.html) and of Chapters [8](ch8.html) and [9](ch9.html). And the mean return, with every indicator built on it, is estimated with an error that more frequent data cannot reduce.
