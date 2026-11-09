<!--
Course text of the web version, Chapter 10 (EN). Inserted into the page by latex/build_web_lecture.py.
Format: # section title (introduction), ## slide title {h=hash} (text), ## @end (summary).
Numbered equations: $$ ... $$ {#eq:label}, reference @eq:label. Citations: @Key -> Author (year), [@Key] -> (Author, year).
Current hashes: python3 build_web_lecture.py 10 --hashes
-->

## @intro

@text
A risk manager has to answer two questions every day: how much can the portfolio lose tomorrow, and was yesterday's answer right? Value at Risk (VaR) answers the first with a quantile of the loss distribution, expected shortfall (ES) with the average loss beyond that quantile, and backtesting answers the second by comparing the forecasts with the losses that actually occurred. Since 1996 these three tools have also been the language of bank regulation: the Basel rules set market-risk capital from a bank's own VaR or ES and raise it when the model fails its backtest.

The chapter builds the tools on daily data from 2000 to 2026 and judges them in two storms, the financial crisis of 2008 and the COVID-19 crash of 2020. On the S&P 500, the historical VaR 1% is 3.46% and the ES 2.5% is 3.76%, while a Normal distribution fitted to the same returns gives 2.80% and 2.81%. From September 2008 to March 2009, a rolling Normal VaR 1% of the S&P 500 was exceeded on 21 of 146 days, where a correct model would have produced about 1.5 exceptions; a VaR built on a GARCH volatility forecast was exceeded on 5 days.

The chapter therefore combines the two previous strands of the course: the tails of [Chapter 5](ch5.html) and the volatility forecasts of [Chapter 9](ch9.html). It shows that the choice of tail distribution matters less than tracking today's volatility, that correlation understates the risk of joint crashes, and that one year of data separates a good risk model from a bad one only roughly.

@chart sfm_ch10_rolling_var
Daily log returns of the S&P 500 (%) in 2007–2009 (left) and 2019–2021 (right), with minus the VaR 1% forecasts of four methods: historical simulation (HS) and the Normal distribution on a 500-day window, GARCH-t and filtered historical simulation (FHS); dots mark the days on which the HS VaR was exceeded.

@goals
- **Define** VaR and ES at a tail probability $\alpha$ (VaR 1%, ES 2.5%) and compute them for the Normal and the Student-t distribution.
- **State** the axioms of a coherent risk measure and show with a counterexample that VaR is not subadditive.
- **Estimate** VaR and ES by historical simulation, parametric methods, Cornish–Fisher, EVT, GARCH and filtered historical simulation.
- **Compute** the VaR of a portfolio and explain why correlation does not measure the risk of joint crashes.
- **Backtest** a VaR model with the Kupiec and Christoffersen tests and with the Basel traffic light.

@plan
- Measuring Tail Risk | The 4:15 report, RiskMetrics, the Basel rules and the two crises used as stress tests.
- VaR and ES: Definitions | Losses, horizon and level; VaR as a quantile, ES as a tail average; closed forms for the Normal and the Student-t; the square-root-of-time rule.
- Coherent Risk Measures | The four axioms, a counterexample in which VaR penalises diversification, and elicitability.
- Estimating VaR and ES | Historical simulation and its precision, the Normal, Student-t, Cornish–Fisher and EVT methods on six series.
- Conditional VaR: GARCH and Filtered Historical Simulation | VaR driven by today's volatility, rolling forecasts and the crises of 2008 and 2020.
- Portfolio VaR and Dependence | The variance–covariance method, correlation in crises, tail dependence and copulas.
- Backtesting | The hit sequence, the Kupiec and Christoffersen tests, the Basel traffic light and the backtest of ES.
- AI for Scientific Discovery | An open question on tail dependence between Bucharest and world markets, the role of AI and the checks.
- Summary | Key results, formulas and self-check questions.

# Measuring Tail Risk

The section tells how a single daily number came to summarise the risk of a bank. It starts with the request of J.P. Morgan's chairman for a daily report, follows VaR from the RiskMetrics document of 1994 to the Basel capital rules of 1996 and 2019, and introduces the two crises, 2008 and 2020, in which every method of the chapter is tested.

## The 4:15 Report and RiskMetrics {h=4c634176}

Around 1990, Dennis Weatherstone, the chairman of J.P. Morgan, asked for a single number, delivered every day at 4:15 pm after the close of trading, that would say how much the bank could lose over the next day. The bank's trading desks held positions in bonds, currencies, equities and derivatives, and no individual report could add them up. The answer became the **Value at Risk** (VaR): a loss that is exceeded only with a small probability over a given horizon.

In October 1994 the bank published its methodology and the data needed to apply it free of charge, under the name RiskMetrics; the fourth edition of the technical document is @JPMorganReuters1996. The method was deliberately simple: Normal returns, a volatility estimated by an exponentially weighted moving average (EWMA, the estimator of [Chapter 8](ch8.html)) and a VaR at a tail probability of 5% over one day.

The publication turned VaR into a common language. Banks, regulators and fund managers could now state their market risk in the same units, a loss in currency at a given probability, and compare positions that had nothing else in common. The simplicity that made the method popular also contains its two weak points, which the chapter examines in turn: the Normal distribution has too thin a tail, and a quantile says nothing about the size of the losses beyond it.

## RiskMetrics and the Basel Rules {h=e4a6ffa4}

The Basel Committee on Banking Supervision (BCBS), hosted by the Bank for International Settlements (BIS) in Basel, sets the international standards for bank capital ([history of the Basel Committee](https://www.bis.org/bcbs/history.htm)). In 1996 it allowed banks to compute the capital for market risk with their own internal VaR models [@BCBS1996a]. The rule uses VaR 1% over 10 days, multiplied by a factor of at least 3, and the factor increases when the model fails its backtest, that is, when actual losses exceed the VaR too often [@BCBS1996b].

In 2019 the Fundamental Review of the Trading Book (FRTB) replaced VaR 1% with ES 2.5% as the basis of the capital charge [@BCBS2019]; the current rule is chapter MAR33 of the Basel Framework [@BCBS2023b]. The change answers the main criticism of VaR: a quantile ignores how bad the losses beyond it are, while ES averages them.

Backtesting did not move to ES. Under chapter MAR32 of the Basel Framework [@BCBS2023a], banks still count the days on which the loss exceeded the VaR forecast, at the level of the bank and of each trading desk. The reason is statistical and appears in the [Coherent Risk Measures section](#sec-3): a VaR forecast can be checked against realised losses on its own, while an ES forecast can only be checked together with a VaR forecast.

## The Crises of 2008 and 2020 {h=f999af02}

The photographs show the two crises that frame the chapter: the headquarters of Lehman Brothers in New York on 15 September 2008, the day of its bankruptcy, and the deserted financial district of New York on 25 March 2020, during the COVID-19 lockdown. The worst day of the S&P 500 since 2000 came in the second crisis: a daily log return of −12.8% on 16 March 2020.

A Normal model with the sample volatility gives such a day a probability that is practically zero; [Chapter 5](ch5.html) measured how far from zero. A risk measure built on that model would have been useless exactly when it was needed.

The chapter therefore uses September 2008 – March 2009 and February – May 2020 as stress tests for every method. A method that gives sensible numbers in calm years but fails in these two periods does not measure tail risk; it measures ordinary variation.

## @end

VaR was born as a practical answer to a management question and became a regulatory standard within a few years: Basel 1996 based market-risk capital on VaR 1% over 10 days, and the FRTB of 2019 replaced it with ES 2.5% while keeping the count of VaR exceptions as the backtest. The [next section](#sec-2) defines both measures precisely and computes them for the S&P 500.

# VaR and ES: Definitions

The section defines VaR and ES as functions of the distribution of the profit and loss, computes them first on a toy sample of 100 days and then on 26 years of S&P 500 returns, derives the closed forms for the Normal and the Student-t distributions, and examines the rule that scales a one-day VaR to ten days.

## Losses, Horizon and Level {h=18c3a84e}

A risk measure needs three ingredients. The first is the random variable $X$: the profit and loss (P&L) or the return of a position over a horizon $h$; in this chapter $h$ is one day and $X$ is the daily log return in percent. The loss is $L = -X$, a positive number when the position loses money. Both conventions appear in the literature, and most errors in practice come from mixing them.

The second ingredient is the level $\alpha$, the tail probability, for example $\alpha = 1\%$ or $\alpha = 2.5\%$. The course writes **VaR 1%** and **ES 2.5%**: the number in the name is always the probability of the tail. Regulatory texts and software often speak of "VaR 99%", naming the complementary confidence level; the two names describe the same number, but the course uses only the tail probability, so that a smaller number always means a more extreme loss.

The third ingredient is the quantile. The quantile of order $\alpha$ of $X$ is $q_\alpha(X) = \inf\{x : P(X \le x) > \alpha\}$, the definition of [Chapter 2](ch2.html) that also works for discrete distributions. For a continuous distribution function $F$ it is simply $q_\alpha = F^{-1}(\alpha)$. For small $\alpha$ the quantile of returns is a negative number: the return below which the worst $\alpha$ of the days fall.

## Value at Risk {h=66dc560c}

The Value at Risk at level $\alpha$ is the negative of the $\alpha$-quantile of the P&L:

$$ \mathrm{VaR}_\alpha(X) = -q_\alpha(X). $$ {#eq:var}

The minus sign turns the quantile into a positive loss. The definition implies $P(L > \mathrm{VaR}_\alpha) \le \alpha$: the loss exceeds the VaR with probability at most $\alpha$. A daily VaR 1% of 3% therefore says that a loss larger than 3% occurs on about one day in a hundred. In money, a position worth $W$ has a VaR of $W \times \mathrm{VaR}(\%)/100$.

What VaR does not say matters as much. It says nothing about the size of the losses beyond it: two positions with the same 1% quantile, one of which loses 4% on its bad days and the other 40%, have the same VaR. And VaR is not the maximum loss. By construction it is exceeded on about a fraction $\alpha$ of the days; a VaR that is never exceeded is not a prudent VaR but a wrong one, as the backtests of the [Backtesting section](#sec-7) show.

## Expected Shortfall {h=e894f113}

Expected shortfall answers the question VaR leaves open: how large is the loss on the bad days? It is defined as the average of the VaR over all levels below $\alpha$:

$$ \mathrm{ES}_\alpha(X) = -\frac{1}{\alpha}\int_0^\alpha q_u(X)\,du, $$ {#eq:es}

where the integration variable $u$ runs over the tail levels from 0 to $\alpha$. For a continuous distribution the definition reduces to $\mathrm{ES}_\alpha = -E[X \mid X \le q_\alpha(X)]$: the average loss on the worst $\alpha$ of the days. The same quantity appears under other names, conditional VaR (CVaR), tail VaR (TVaR) or expected tail loss; for continuous distributions they coincide.

Two properties follow from the definition. Because the quantiles $q_u$ for $u < \alpha$ all lie below $q_\alpha$, their average does too, so $\mathrm{ES}_\alpha \ge \mathrm{VaR}_\alpha$ at the same level. And ES looks inside the tail: two distributions with the same VaR but different tails, one ending abruptly and one with rare catastrophic losses, have different ES. The integral form @eq:es is the general definition; it remains valid for discrete distributions, where the conditional expectation can give a different and less well-behaved number.

## Worked Example: VaR and ES from 100 Days {h=a2e50fc0}

With a sample, the quantile is replaced by an order statistic. A position has 100 daily returns, and the five worst are −6.1%, −4.8%, −4.2%, −3.9% and −3.5%. The empirical quantile of level $\alpha$ is the $k$-th smallest return, with $k = \lceil n\alpha \rceil$, where $\lceil\cdot\rceil$ rounds up to the next integer.

For VaR 1%, $k = \lceil 100 \times 0.01 \rceil = 1$: the empirical quantile is the worst return, and $\mathrm{VaR}_{1\%} = 6.1\%$. A single observation decides this estimate, so with 100 days it is very imprecise: a different sample of 100 days would almost surely have a different worst day. For ES 2.5%, $k = \lceil 2.5 \rceil = 3$, and the estimate is the average of the three largest losses, $\mathrm{ES}_{2.5\%} = (6.1 + 4.8 + 4.2)/3 = 5.03\%$, while $\mathrm{VaR}_{2.5\%}$ is the third largest loss, 4.2%.

Translated into money, a position of 1 million lei has a VaR 1% of 61,000 lei: on one day in a hundred, the loss is larger. The example also shows why ES at 2.5% and VaR at 1% are natural partners: ES 2.5% uses three observations where VaR 1% uses one, and its value, 5.03%, lies between the worst and the third-worst loss.

## VaR 1% and ES 2.5% of the S&P 500 {h=187cb6b7}

The histogram shows the 6,714 daily log returns of the S&P 500 since 2000. The shaded area is the worst 2.5% of the days. The solid line marks minus the historical VaR 1%, at −3.46%, and the dashed line minus the ES 2.5%, at −3.76%; the historical VaR 2.5%, the right edge of the shaded area, is 2.51%.

The two lines answer different questions. VaR 1% is a single point of the tail: 1% of the days lie to its left, wherever they lie. ES 2.5% averages the whole shaded tail, including the extreme days of 2008 and 2020 that are barely visible as isolated bars far to the left; if those days had been twice as bad, the VaR 1% would be unchanged while the ES would rise.

For a position of 1 million lei in the index, the two numbers translate into a VaR 1% of about 34,552 lei and an ES 2.5% of about 37,613 lei. The ES 2.5% is larger than the VaR 1% even though it refers to a less extreme probability, because it averages the losses beyond the 2.5% quantile, and some of these losses are much larger than 3.46%.

## The Normal Distribution: Closed Forms {h=87ce375e}

If $X \sim N(\mu, \sigma^2)$, both measures have closed forms. With $z_\alpha = \Phi^{-1}(\alpha)$ the standard Normal quantile, where $\Phi$ and $\varphi$ are the distribution function and the density of $N(0, 1)$,

$$ \mathrm{VaR}_\alpha = -(\mu + \sigma z_\alpha), \qquad \mathrm{ES}_\alpha = -\mu + \sigma\,\frac{\varphi(z_\alpha)}{\alpha}. $$ {#eq:normal}

The ES formula rests on the tail mean of the standard Normal, $E[Z \mid Z \le z] = -\varphi(z)/\Phi(z)$. The derivation uses $\varphi'(z) = -z\varphi(z)$: integrating $t\varphi(t)$ from $-\infty$ to $z$ gives $-\varphi(z)$, and dividing by $P(Z \le z) = \Phi(z) = \alpha$ gives the conditional mean.

The constants are worth knowing: $z_{0.01} = -2.326$, $z_{0.025} = -1.960$, $\varphi(z_{0.025}) = 0.0584$, and the ES factor at 2.5% is $\varphi(z_{0.025})/0.025 = 2.338$. With $\mu = 0$, the ratio of ES 2.5% to VaR 1% is $2.338/2.326 = 1.005$: for Normal returns the two measures are practically equal. This is why the FRTB chose ES 2.5% to replace VaR 1%: for Normal data the capital charge stays about the same, and for heavy-tailed data, where the ES 2.5% exceeds the VaR 1%, it rises.

## Worked Example: Normal VaR and ES of the S&P 500 {h=e46268a8}

The sample moments of the daily S&P 500 returns since 2000 are $\hat\mu = 0.025\%$ and $\hat\sigma = 1.21\%$. Inserted into @eq:normal, they give the Normal VaR 1% $= -(0.025 - 2.326 \times 1.21) = -0.025 + 2.822 = 2.80\%$ and the Normal ES 2.5% $= -0.025 + 2.338 \times 1.21 = -0.025 + 2.836 = 2.81\%$.

The data tell a different story. Historical simulation gives VaR 1% = 3.46% and ES 2.5% = 3.76%. The Normal VaR 1% is 19% too small, because the Normal tail is too thin ([Chapter 2](ch2.html)). The Normal ES 2.5% is even further from the data, about a quarter below the historical value, because the Normal tail has nothing like the extreme days that the historical average includes.

Two parameters, a mean and a standard deviation, are fitted to the centre of the distribution, where most of the observations are; they cannot also describe the tail. The methods of the following sections take one of two routes: they change the distribution (Student-t, Cornish–Fisher, EVT) or they change the data used (historical simulation, volatility filtering).

## The Student-t Distribution: Closed Forms {h=fecdc844}

The Student-t distribution of [Chapter 2](ch2.html) has a power-law tail. Write $X = m + s\,T$ with $T \sim t_\nu$, a location $m$, a scale $s$ and $\nu$ degrees of freedom; let $t_\alpha = t_\nu^{-1}(\alpha)$ be the quantile and $f_\nu$ the density of $t_\nu$. Then

$$ \mathrm{VaR}_\alpha = -(m + s\,t_\alpha), \qquad \mathrm{ES}_\alpha = -m + s\,\frac{f_\nu(t_\alpha)}{\alpha}\,\frac{\nu + t_\alpha^2}{\nu - 1}, \quad \nu > 1. $$ {#eq:tdist}

The condition $\nu > 1$ is needed for the mean, and hence the ES, to exist. The scale $s$ is not the standard deviation: the variance of $X$ is $s^2\nu/(\nu - 2)$, so a standardised t with variance 1 has $s = \sqrt{(\nu - 2)/\nu}$. This standardisation reappears in the GARCH-t model of the [conditional VaR section](#sec-5).

Fitted to the S&P 500 by maximum likelihood, the distribution has location $\hat m = 0.068$, scale $\hat s = 0.703$ and $\hat\nu = 2.67$ degrees of freedom. The quantile is $t_{0.01} = -5.023$, so the VaR 1% $= -(0.068 + 0.703 \times (-5.023)) = 3.46\%$, exactly the historical value. The ES factor is 5.720, so the ES 2.5% $= -0.068 + 0.703 \times 5.720 = 3.95\%$, above the historical 3.76%. With $\nu < 3$ the fitted tail is very heavy: it matches the 1% quantile, but beyond it the t distribution places more mass than the data show.

## Scaling VaR from One to Ten Days {h=2d80107b}

Regulatory VaR has a 10-day horizon, while data and models are daily. The **square-root-of-time rule** bridges the gap: $\mathrm{VaR}^{(h)} \approx \sqrt{h}\,\mathrm{VaR}^{(1)}$. The rule is exact for i.i.d. Normal returns with mean zero, because the $h$-day return is then Normal with standard deviation $\sigma\sqrt{h}$ and every quantile scales by $\sqrt{h}$. Basel 1996 asked for a 10-day VaR, usually obtained from the one-day VaR by the factor $\sqrt{10} = 3.162$.

For the S&P 500, the historical one-day VaR 1% is 3.46%, and the rule gives $3.162 \times 3.46 = 10.93\%$ over 10 days. The historical VaR 1% of overlapping 10-day returns is 10.37%, somewhat smaller: the ratio is 3.00, below $\sqrt{10}$. The difference is not an accident of the sample but the result of three features of real returns.

The first is heavy tails. The sum of 10 heavy-tailed returns is closer to Normal than each of them (the central limit theorem, CLT), so its kurtosis is lower and the ratio of its 1% quantile to the daily one is below $\sqrt{10}$. The second is volatility clustering ([Chapter 9](ch9.html)): after a quiet day volatility tends to rise back to its long-run level, so the 10-day risk is more than $\sqrt{10}$ times today's risk; after a turbulent period it tends to fall, and the 10-day risk is less. The third is autocorrelation. The variance of a 10-day sum is $\sigma^2\bigl(10 + 2\sum_{k=1}^{9}(10-k)\rho_k\bigr)$, so a negative first-order autocorrelation lowers it and a positive one raises it. The S&P 500 has $\hat\rho_1 = -0.10$: consecutive days partly offset each other, which adds to the effect of the heavy tails. The BET has $\hat\rho_1 = +0.11$ (the non-synchronous trading of its constituents), and its ratio is 3.86, above $\sqrt{10}$. The rule is a convention of regulation, not a property of the data.

## @end

VaR @eq:var is a quantile of the loss and ES @eq:es the average loss beyond it. On the S&P 500 since 2000 the historical values are VaR 1% = 3.46% and ES 2.5% = 3.76%; the Normal distribution gives 2.80% and 2.81%, while a Student-t matches the VaR and overshoots the ES. The [next section](#sec-3) asks which properties a good risk measure should have and finds that VaR lacks one of them.

# Coherent Risk Measures

The section judges VaR and ES by two theoretical criteria. The first, coherence, asks whether a risk measure behaves sensibly when positions are scaled or combined; the four axioms of Artzner, Delbaen, Eber and Heath define it, and a counterexample with two bonds shows that VaR can violate it. The second, elicitability, asks whether forecasts of the measure can be compared with outcomes, and here the ranking of the two measures reverses.

## 1999: Four Axioms for a Risk Measure {h=a25aeac3}

@ArtznerEtAl1999 define a risk measure $\rho$ as a map from a loss $L$ to the amount of capital that makes the position acceptable to a regulator. They call $\rho$ **coherent** if it satisfies four axioms. Monotonicity: if $L_1 \le L_2$ in every state of the world, then $\rho(L_1) \le \rho(L_2)$. Translation invariance: $\rho(L + c) = \rho(L) + c$ for a certain amount $c$, so adding a sure loss raises the capital by exactly that amount. Positive homogeneity: $\rho(\lambda L) = \lambda\rho(L)$ for $\lambda > 0$, so doubling a position doubles its capital. Subadditivity: $\rho(L_1 + L_2) \le \rho(L_1) + \rho(L_2)$.

The first three axioms are almost uncontroversial. Subadditivity carries the economic content: merging two positions never creates risk, so diversification never adds risk. It has a practical consequence for a bank: the sum of the capital computed separately for each division is an upper bound for the capital of the whole bank, and the capital can be allocated division by division. A measure that is not subadditive could make a merged bank look riskier than its two halves, and it would reward splitting a portfolio artificially to lower the capital.

The photograph shows Freddy Delbaen, one of the four authors, in 1997. The axioms were published in 1999 in *Mathematical Finance* and changed the question about risk measures from "which number is intuitive?" to "which properties must the number have?".

## Counterexample: Two Bonds, Step by Step {h=21c24abc}

Two independent bonds each lose 100 if the issuer defaults, which happens with probability 4%, and 0 otherwise; the level is $\alpha = 5\%$. For one bond, $P(L > 0) = 4\% \le 5\%$, so the 5% quantile of the loss is 0 and $\mathrm{VaR}_{5\%} = 0$ for each bond. A risk manager using VaR would see no risk in either bond.

For the portfolio of both bonds, the loss is 200 if both default, with probability $0.04^2 = 0.16\%$, and 100 if exactly one defaults, with probability $2 \times 0.04 \times 0.96 = 7.68\%$. Now $P(L > 0) = 7.84\% > 5\%$, so the VaR of the portfolio is 100. Since $\mathrm{VaR}_{5\%}(L_1 + L_2) = 100 > 0 + 0$, VaR penalises diversification: holding both bonds appears riskier than the sum of holding each.

ES 5% is the average of the worst 5% of the outcomes. For one bond, the worst 5% consist of the 4% of default states with loss 100 and 1% of the states with loss 0, so $\mathrm{ES}_{5\%} = (4\% \times 100 + 1\% \times 0)/5\% = 80$. For the portfolio, the worst 5% consist of the 0.16% with loss 200 and 4.84% with loss 100, so $\mathrm{ES}_{5\%} = (0.16\% \times 200 + 4.84\% \times 100)/5\% = 103.2$. Since $103.2 \le 80 + 80$, ES is subadditive here, and it sees the default risk of each bond, which VaR ignores entirely.

## Coherence of ES and VaR {h=1d01080d}

ES satisfies all four axioms [@AcerbiTasche2002]. The reason is visible in @eq:es: ES is an average of quantiles over the tail, and averages respect sums. More precisely, ES can be written as the largest expected loss over a set of reweighted scenarios, and the maximum of a sum is never larger than the sum of the maxima.

VaR satisfies the first three axioms; subadditivity can fail. It fails for discrete losses, such as credit losses and the bonds of the previous example, and for extremely heavy tails, with a tail index below 1 ([Chapter 5](ch5.html)), where the mean loss is infinite. It holds for elliptical distributions such as the Normal and the Student-t, and @DanielssonEtAl2013 show that in practice it holds for most equity returns, whose tail index lies well above 1.

ES pays for coherence with a stronger requirement: it needs a finite mean loss, while VaR exists for every distribution. For daily equity returns, whose tail index is around 3, this is not a constraint; for some operational or insurance losses it can be.

## Elicitability and Forecast Evaluation {h=efb3b356}

A statistic is **elicitable** if some loss function, or scoring function, is minimised in expectation by its true value [@Gneiting2011]. The mean is elicitable through the squared error, the median through the absolute error. Elicitability matters because it allows competing forecasts to be ranked: the forecaster with the smaller average score is the better one, and no forecaster can lower the expected score by reporting anything other than the true value.

VaR is elicitable. The scoring function is the quantile, or pinball, loss

$$ S(v, x) = (\alpha - \mathbf{1}\{x < -v\})(x + v), $$ {#eq:pinball}

where $v$ is the VaR forecast, $x$ the realised return and $\mathbf{1}\{\cdot\}$ equals 1 if the event occurs and 0 otherwise. Its expectation is minimised where $P(X < -v) = \alpha$, that is at $v = \mathrm{VaR}_\alpha$. Two VaR models can therefore be compared by the average quantile loss of their forecasts, as the [Backtesting section](#sec-7) does.

ES alone is not elicitable [@Gneiting2011]: no scoring function is minimised in expectation by the true ES. The pair (VaR, ES), however, is jointly elicitable [@FisslerZiegel2016]. The consequence is practical: forecasts of ES are tested together with forecasts of VaR, as in the ES backtest at the end of the [Backtesting section](#sec-7). Coherence favours ES, elicitability favours VaR, and the Basel rules use both: ES for capital, VaR exceptions for the backtest.

## @end

ES is coherent; VaR is not always subadditive and can penalise diversification, as the two-bond example shows with a portfolio VaR of 100 against individual VaRs of 0. VaR is elicitable through the pinball loss @eq:pinball, while ES is elicitable only together with VaR. The [next section](#sec-4) turns from properties to estimation.

# Estimating VaR and ES

The section compares five unconditional methods: historical simulation, the Normal and Student-t distributions, the Cornish–Fisher expansion and EVT. It computes each step by step for the S&P 500, measures the sampling error of a historical VaR with the bootstrap, and compares all methods on six series: the S&P 500, the DAX and the BET from 2000, Bitcoin from 2014, and two stocks of the Bucharest Stock Exchange (BVB), Banca Transilvania (TLV) and OMV Petrom (SNP), from 2010.

## Summary of the Five Unconditional Methods {h=c0834701}

The table lists the five methods with their assumption and their weak point. Historical simulation (HS) assumes that the past window is a sample from tomorrow's distribution; it has few points in the tail and cannot produce a loss beyond the worst day in the window. The Normal method assumes Normal returns and has tails that are far too thin. The Student-t method allows heavy tails but forces one shape on both tails. The Cornish–Fisher method corrects the Normal quantile for skewness and kurtosis and breaks down when the kurtosis is large. EVT fits a generalised Pareto distribution (GPD) above a high threshold, as in [Chapter 5](ch5.html), and its weak point is the choice of the threshold.

"Unconditional" means that the method gives the same VaR on every day of the estimation window: it describes the distribution of returns averaged over calm and turbulent periods. The conditional methods of the [next section](#sec-5) instead scale the forecast with today's volatility.

All methods use daily log returns in percent ([Chapter 1](ch1.html)) and report VaR and ES as positive losses. This common convention makes the numbers directly comparable across methods and series.

## Historical Simulation {h=048b52ec}

Historical simulation is the simplest estimator. Take the last $n$ returns $r_{t-n+1}, \dots, r_t$ and sort them, $r_{(1)} \le \dots \le r_{(n)}$. Then

$$ \widehat{\mathrm{VaR}}_\alpha = -r_{(k)}, \quad k = \lceil n\alpha \rceil, \qquad \widehat{\mathrm{ES}}_\alpha = -\frac{1}{k}\sum_{i=1}^{k} r_{(i)}. $$ {#eq:hs}

No distribution is assumed: the empirical distribution of the window is the forecast. For the S&P 500 since 2000, $n = 6{,}714$. The VaR 1% uses $k = \lceil 67.14 \rceil = 68$, and the 68th worst day is a loss of 3.46%. The ES 2.5% is the average of the worst $\lceil 167.85 \rceil = 168$ days, 3.76%.

The window length $n$ is a trade-off. A short window of 250 days reacts to recent events, but its VaR 1% is the third worst return, $k = \lceil 2.5 \rceil = 3$, and rests on three observations. A long window, here all the data, gives a stable estimate that does not react to today's volatility at all.

A specific artefact of a fixed window is the **ghost effect**. A crash day changes the VaR when it enters the window, which is reasonable, and changes it again, abruptly, $n$ days later when it leaves the window, although nothing happened on that day. With a 500-day window, a crash stays in the estimate for about two years, as the rolling forecasts of the [conditional VaR section](#sec-5) show.

## Precision of a Historical VaR {h=2f738ca3}

A VaR estimate is a statistic and has a sampling error. The bootstrap of [Chapter 2](ch2.html) measures it: resample the $n$ returns with replacement 2000 times, recompute VaR and ES on each resample, and read the 2.5% and 97.5% quantiles of the 2000 values as a 95% interval.

With all 6,714 S&P 500 days, the VaR 1% of 3.46% has the 95% interval [3.18%, 3.66%], and the ES 2.5% of 3.76% has [3.46%, 4.07%]. With the last 500 days only, the VaR 1% is 2.75% with the interval [1.99%, 4.96%], and the ES 2.5% is 2.94% with [2.14%, 3.82%]. With two years of data, the interval for the VaR 1% is almost three percentage points wide, because the estimate rests on the five worst days, $k = \lceil 500 \times 0.01 \rceil = 5$. The interval of the ES is narrower, since an average of 13 observations varies less than a single order statistic.

The intervals set a scale for every comparison that follows. A difference between two methods that is smaller than the bootstrap interval does not show that one method is better; it may be sampling noise. The resampling treats the days as independent, so with volatility clustering the true uncertainty is, if anything, larger.

## The Cornish–Fisher Expansion {h=eab869dc}

@CornishFisher1938 correct the Normal quantile $z = z_\alpha$ with the skewness $S$ and the excess kurtosis $K$, the kurtosis coefficient minus 3:

$$ z_{CF} = z + \frac{(z^2 - 1)S}{6} + \frac{(z^3 - 3z)K}{24} - \frac{(2z^3 - 5z)S^2}{36}, \qquad \mathrm{VaR}_\alpha = -(\mu + \sigma z_{CF}). $$ {#eq:cf}

The expansion is a series in the higher cumulants around the Normal quantile. At $z_{0.01} = -2.326$, a negative skewness $S$ and a positive excess kurtosis $K$ both push the quantile further into the left tail, so the VaR rises above the Normal value.

For the S&P 500, $S = -0.35$ and $K = 10.7$. The kurtosis term alone is $(z^3 - 3z)K/24 \approx -2.50$, larger than the Normal quantile itself, and the corrected quantile is $z_{CF} = -5.03$, which gives a VaR 1% of 6.08%, against the historical 3.46%. The correction has overshot by more than the Normal method undershot.

The expansion is a small-correction device: it is accurate when the distribution is close to Normal and the correction terms are small compared with $z$. With an excess kurtosis above 10, the terms are no longer small and the expansion lies far above the correct value. The method is useful only when $|S|$ and $K$ are moderate, for example for monthly returns or for diversified portfolios.

## EVT: Peaks over Threshold (Chapter 5) {h=a33b0c19}

The peaks-over-threshold (POT) method of [Chapter 5](ch5.html) models only the tail. With the losses $L = -r$ and a threshold $u$ set at the 90% quantile of the losses, as in @McNeilFrey2000, the excesses $L - u$ above $u$ follow approximately a GPD with shape $\xi$ and scale $\beta$. The tail estimator then gives

$$ \mathrm{VaR}_\alpha = u + \frac{\beta}{\xi}\Big[\Big(\frac{n\alpha}{N_u}\Big)^{-\xi} - 1\Big], \qquad \mathrm{ES}_\alpha = \frac{\mathrm{VaR}_\alpha + \beta - \xi u}{1 - \xi}, $$ {#eq:pot}

where $n$ is the number of days and $N_u$ the number of excesses. The ratio $n\alpha/N_u$ compares the target tail probability with the share of days above the threshold; a shape $\xi > 0$ means a heavy, Pareto-type tail.

For the S&P 500, $u = 1.26\%$, $N_u = 672$ excesses (10.0% of the days), $\hat\xi = 0.164$ and $\hat\beta = 0.820$. For VaR 1%, $n\alpha/N_u = 0.0999$ and $0.0999^{-\hat\xi} = 1.461$; with $\hat\beta/\hat\xi = 4.988$, the VaR 1% is $1.26 + 4.988 \times (1.461 - 1) = 3.56\%$. The ES 2.5% is 3.77%. Both are close to the historical values, 3.46% and 3.76%.

At 1% and 2.5% the EVT and historical estimates agree because both are driven by the same few hundred tail observations. EVT adds value further out: it smooths the tail with a two-parameter model and extrapolates beyond the worst observed day, which makes it the method for VaR 0.1% and for stress tests.

## VaR across Tail Probabilities {h=039c9bf2}

The chart plots, for the S&P 500 (left) and the BET (right), the VaR of each method as a function of the tail probability $\alpha$, from 5% on the left to 0.1% on the right, on a logarithmic axis; the dots are the historical estimates. Reading from left to right means moving deeper into the tail.

At 5% all methods agree: so close to the centre, the shape of the tail hardly matters. At 1% the Normal line already lies below the dots, and the gap widens as $\alpha$ falls, because the Normal quantile grows only like $\sqrt{2\ln(1/\alpha)}$. At 0.1% the Student-t and EVT lines stay close to the data, while the Cornish–Fisher line rises explosively, to roughly twice the historical value.

The chart summarises the section in one picture. Methods with a power-law tail, the Student-t and EVT, follow the data across the whole range of levels; the Normal method fails more the further one goes into the tail; and Cornish–Fisher, built for small departures from the Normal, fails in the opposite direction.

## VaR 1% by Method on Six Series {h=d88f8fdc}

The table reports VaR 1% in percent for six series over their full samples (the S&P 500, DAX and BET from 2000, Bitcoin from 2014, TLV and SNP from 2010), together with the excess kurtosis $K$ and the GPD shape $\hat\xi$. The levels differ widely, from 3.46% for the S&P 500 to 10.54% for Bitcoin, but the pattern across methods is the same in every row.

Historical simulation, Student-t and EVT differ by at most 0.7 percentage points; the largest gap is for Bitcoin, where the Student-t gives 11.06% and EVT 10.38%. Given the bootstrap intervals of the previous slides, these three methods agree. The Normal VaR is too small everywhere, by 14% to 24%: from 2.80% against 3.46% for the S&P 500 to 7.99% against 10.54% for Bitcoin.

Cornish–Fisher gives 1.3 to 2.0 times the historical VaR, from 5.42% against 4.20% for the DAX, which has the lowest excess kurtosis (6.1), to 10.19% against 5.11% for TLV, which has the highest (13.5). With $K$ between 6 and 14, the expansion is outside its range of validity for every series; the overshoot is smallest for the DAX and largest for TLV.

## ES 2.5% by Method on Six Series {h=e84d7f0f}

The same comparison for ES 2.5% shows how much the tail beyond the quantile differs between methods. Under the Normal distribution, ES 2.5% equals VaR 1% almost exactly (2.81% against 2.80% for the S&P 500). In the data, the historical ES 2.5% lies between about 1.5% and 10.5% above the historical VaR 1% (computed from the unrounded values), depending on the series: the heavier the tail beyond the 1% quantile, the larger the gap.

For the S&P 500, the DAX, the BET and Bitcoin, the Student-t gives the largest ES among the reasonable methods; for Bitcoin it is 13.37% against the historical 10.84%. With $\hat\nu < 3$, the fitted t has a heavier tail than the data beyond the threshold: one parameter fits the 1% quantile well, but it then extrapolates too much mass into the extreme tail.

EVT and historical simulation agree at 2.5% for every series, to within 0.06 percentage points. The choice of method therefore matters more for ES than for VaR, because ES depends on the whole tail, while VaR depends on one point of it. This is the price of the coherence of ES: it is more sensitive to the model of the far tail.

## @end

For VaR 1% and ES 2.5%, historical simulation, the Student-t and EVT agree on all six series, while the Normal VaR is 14–24% too small and Cornish–Fisher overshoots by a factor of 1.3 to 2.0. The bootstrap shows that a historical VaR 1% on two years of data is uncertain by several percentage points. All five methods, however, give the same VaR on every day of the window. The [next section](#sec-5) makes the forecast depend on today's volatility.

# Conditional VaR: GARCH and Filtered Historical Simulation

The section combines the volatility forecasts of [Chapter 9](ch9.html) with the tail models of the previous section. A VaR forecast becomes the product of tomorrow's volatility and a quantile of the standardised innovation. The section computes such forecasts for six series, evaluates rolling forecasts from 2005 to 2026, and compares conditional and unconditional methods in the crises of 2008 and 2020.

## Conditioning on Current Volatility {h=2ccc22e0}

Volatility clusters ([Chapters 8](ch8.html) and [9](ch9.html)): calm days follow calm days and turbulent days follow turbulent days. An unconditional VaR averages over both regimes, so it is too high in quiet years and too low at the start of a crisis, exactly when it matters most.

The conditional model writes tomorrow's return as $r_{t+1} = \mu + \sigma_{t+1}z_{t+1}$, where $\sigma_{t+1}$ is the volatility forecast made today and $z$ is an i.i.d. innovation with mean 0 and variance 1. Because the only time-varying part is the scale, the risk measures follow from those of $z$:

$$ \mathrm{VaR}_{t+1} = -(\mu + \sigma_{t+1}\,q_\alpha(z)), \qquad \mathrm{ES}_{t+1} = -\mu + \sigma_{t+1}\,\mathrm{ES}_\alpha(z), $$ {#eq:cond}

where $q_\alpha(z)$ and $\mathrm{ES}_\alpha(z)$ are the quantile and the ES of the standardised innovation. The model has two ingredients: a forecast of $\sigma_{t+1}$ and a distribution for $z$.

The chapter uses a GARCH(1,1) model [@Bollerslev1986] for the volatility and two variants for $z$. **GARCH-t** assumes a standardised Student-t, with $\nu$ estimated jointly with the GARCH parameters. **Filtered historical simulation** (FHS), proposed by @HullWhite1998 and by @BaroneAdesiGiannopoulosVosper1999, takes the empirical distribution of the standardised residuals $\hat z_t = (r_t - \hat\mu)/\hat\sigma_t$: the volatility comes from the model, the shape of the tail from the data.

## Worked Example: the S&P 500 on 18 September 2026 {h=c1448a50}

A GARCH(1,1)-t model fitted to all S&P 500 data gives $\hat\mu = 0.079$, $\hat\alpha = 0.123$, $\hat\beta = 0.872$ and $\hat\nu = 6.26$; the persistence $\hat\alpha + \hat\beta = 0.995$ is typical of daily equity returns. The forecast for the day after 18 September 2026 is $\sigma_{t+1} = 0.714\%$.

For GARCH-t, the quantile of the standardised t is $q_{0.01}(z) = t_\nu^{-1}(0.01)\sqrt{(\nu - 2)/\nu} = -2.557$ and its ES 2.5% is 2.644. By @eq:cond, the VaR 1% is $-(0.079 + 0.714 \times (-2.557)) = 1.75\%$ and the ES 2.5% is $-0.079 + 0.714 \times 2.644 = 1.81\%$. For FHS, the 1% quantile of the 6,714 standardised residuals is −2.787 and their ES 2.5% is 2.952, so the VaR 1% is $-(0.079 + 0.714 \times (-2.787)) = 1.91\%$ and the ES 2.5% is 2.03%.

The market was quiet: the conditional VaR 1% is about half the unconditional historical value of 3.46%. The two variants differ by the tail of $z$: the empirical residuals have a heavier left tail (quantile −2.787) than the fitted t (−2.557), because a single $\nu$ must fit both tails, while the losses of the S&P 500 are more extreme than its gains. FHS keeps this asymmetry.

## Next-Day VaR and ES on Six Series {h=18a12263}

The table gives the forecasts for the day after 18 September 2026 for the six series: VaR 1% and ES 2.5% by unconditional historical simulation on all data, HS (all), by GARCH-t and by FHS, together with the volatility forecast $\sigma_{t+1}$ in percent.

For the S&P 500, the DAX and Bitcoin the period is quiet and the conditional values are below the unconditional ones: the DAX VaR 1% is 2.31% (GARCH-t) and 2.48% (FHS) against 4.20% historically, and Bitcoin's is 7.37% and 7.99% against 10.54%. For the BET, TLV and SNP the current volatility is high, $\sigma_{t+1}$ is 1.763% for the BET, 2.791% for TLV and 1.966% for SNP, and the conditional VaR lies above the long-run historical VaR: for TLV 7.18% (GARCH-t) and 7.33% (FHS) against 5.11%.

The same method thus sends opposite messages on different markets, and this is the point of conditioning. An unconditional VaR would tell a holder of TLV that the risk is the long-run average, when the current volatility is higher than usual, and would tell a holder of the S&P 500 the same, when it is lower. In every row FHS gives a slightly higher VaR and ES than GARCH-t, for the reason seen on the previous slide: the empirical residuals have a heavier left tail than the symmetric t.

## Rolling Forecasts: the Design {h=1af1eefd}

A fair comparison of methods requires out-of-sample forecasts: the forecast for day $t$ uses only data up to day $t-1$, as a risk manager would have had to do in real time. The evaluation covers every day from January 2005, which includes 2008, for the S&P 500, the DAX and the BET, and from January 2017 for Bitcoin.

Four methods are compared. HS and the Normal method use a rolling window of the last 500 days, about two years. GARCH-t and FHS re-estimate the GARCH(1,1)-t model every 250 days on all past data, while the volatility $\sigma_t$ is updated every day with the latest return. Re-estimating the parameters once a year is enough, because they change slowly; updating the volatility daily is essential, because it changes quickly.

Each day, every method produces three numbers: VaR 1%, used in the backtests of VaR, and VaR 2.5% and ES 2.5%, used in the backtest of ES. The resulting sequences are the material of the rest of the chapter.

## VaR 1% through 2008 and 2020 {h=f2ae1841}

The chart shows the S&P 500's daily returns (light line) in 2007–2009 and 2019–2021, with minus the VaR 1% forecasts of the four methods; the dots mark the days on which the HS VaR was exceeded. A return below a line is an exception for that method.

HS and the Normal method react with a delay of months. Their 500-day window changes slowly: a few crisis days move the 1% quantile of 500 observations only a little, so in September–October 2008 and in March 2020 the dots cluster where the returns fall through the flat dashed lines. GARCH-t and FHS, whose lines almost coincide, drop sharply within days, because each large return raises tomorrow's volatility forecast.

After the crisis the behaviour reverses. The HS VaR stays high for two years, until the crisis days leave the window, the ghost effect of the historical method, while the GARCH-t VaR returns to normal levels as volatility decays. A bank using HS would have held too much capital in 2009–2010 and too little in the autumn of 2008.

## Stress Periods: Exceptions in 2008 and 2020 {h=4ed01134}

The table counts the exceptions of the VaR 1%, the days with $r_t < -\mathrm{VaR}_t$, in the two stress periods, September 2008 – March 2009 and February – May 2020; the column "expected" is 1% of the days of the period. In 2008 the S&P 500 had 146 trading days in the period, so a correct VaR 1% would have been exceeded about 1.5 times; the Normal VaR was exceeded on 21 days, HS on 15, and GARCH-t and FHS on 5 each.

The pattern holds for all three indices and both crises. In 2008–2009 the Normal method had 18 exceptions for the DAX and 14 for the BET, against 3 and 4 for the conditional methods; in 2020, with about 0.7 exceptions expected, the Normal method had 12, 13 and 10 for the S&P 500, the DAX and the BET, and GARCH-t and FHS 3, 3 and 4.

No method is perfect in a crisis: even GARCH-t and FHS exceed their expected count by a factor of 2 to 6, because volatility jumps faster than any model of yesterday's returns can follow. But the conditional methods fail far less often, and GARCH-t and FHS, which differ only in the tail of $z$, have identical counts in every cell. The forecast of volatility matters more than the shape of the tail.

## Choosing a VaR Method {h=8571c68a}

The table matches situations with methods. For a daily VaR of a liquid position, GARCH-t and FHS track volatility, and FHS keeps the empirical tail of the standardised residuals. For a quick check on a long history, HS needs no model and is easy to explain. For very small tail probabilities, such as 0.1%, and for stress tests, EVT extrapolates beyond the worst observed day. For portfolios of many assets, the variance–covariance method or a copula model the dependence, the t copula when joint crashes matter ([next section](#sec-6)). For moderate skewness and kurtosis, Cornish–Fisher is a quick correction of the Normal.

Two rules hold whatever the situation. The Normal distribution alone is never adequate for daily data: it underestimates the 1% quantile by 14–24% on every series of the chapter. And a VaR number is meaningless without its method, its window and its level, which must always be reported with it.

Every method, finally, has to be checked against the losses that actually occurred. The backtests of the [Backtesting section](#sec-7) do this for the rolling forecasts of this section.

## @end

Conditional VaR @eq:cond scales a quantile of the standardised innovation by tomorrow's volatility. On 18 September 2026 it gives a VaR 1% of the S&P 500 of 1.75% (GARCH-t) and 1.91% (FHS), about half the unconditional value, while for the BET, TLV and SNP it lies above the long-run value. In the crises of 2008 and 2020, the conditional methods had 3 to 5 exceptions where the Normal method had 10 to 21. The [next section](#sec-6) moves from a single series to a portfolio.

# Portfolio VaR and Dependence

The section computes the VaR of a portfolio of two indices, the BET and the S&P 500. The variance–covariance method combines the volatilities through the correlation; the data then show that correlation rises in crises and that it does not measure the probability of joint crashes. Copulas separate the dependence from the individual distributions, and a t copula captures joint crashes that a Gaussian copula misses.

## A BET and S&P 500 Portfolio {h=81a2b038}

The portfolio holds equal weights, $w = (0.5, 0.5)$, in the BET and the S&P 500. It uses simple returns $R_t$ in percent rather than log returns, because the simple return of a portfolio is exactly the weighted sum of the simple returns of its assets, $R_{p,t} = w_1R_{1,t} + w_2R_{2,t}$; for log returns this holds only approximately. Prices are first aligned on the 6,466 trading days common to both markets since 2000, and the returns are computed afterwards, so that a holiday in one market does not create a spurious two-day return in the other.

Each index is taken in its own currency, so the currency risk of an investor holding the portfolio in lei is ignored. The daily moments are $\hat\mu = 0.076\%$ and $\hat\sigma = 1.42\%$ for the BET, $\hat\mu = 0.034\%$ and $\hat\sigma = 1.23\%$ for the S&P 500, and their correlation is $\hat\rho = 0.21$.

The low correlation partly reflects trading hours. The sessions of the two exchanges overlap only partly (the BVB trades until 17:45 Romanian time, New York opens at 16:30): most of the New York session takes place after the BVB has closed, so part of the common reaction to world news appears in the BET only on the next day, and the same-day correlation understates the link between the two markets. The question of the section is whether the portfolio VaR is below the average of the two individual VaRs, and by how much.

## The Variance–Covariance Method {h=5cd0950c}

With the weight vector $w$, the vector of mean returns $\mu$ and the covariance matrix $\Sigma$, the portfolio has mean $\mu_p = w^\top\mu$ and variance

$$ \sigma_p^2 = w^\top\Sigma w = w_1^2\sigma_1^2 + w_2^2\sigma_2^2 + 2w_1w_2\rho\sigma_1\sigma_2, $$ {#eq:portvar}

where $\rho$ is the correlation of the two assets. If the returns are jointly Normal, the portfolio return is Normal too and $\mathrm{VaR}_p = -(\mu_p + z_\alpha\sigma_p)$; this is the RiskMetrics method [@JPMorganReuters1996], which scales to hundreds of assets because it needs only a covariance matrix.

Step by step: $w_1\sigma_1 = 0.712$, $w_2\sigma_2 = 0.614$ and the covariance term is $2 \times 0.5 \times 0.5 \times 0.369 = 0.185$, where $0.369 = \rho\sigma_1\sigma_2$; hence $\sigma_p^2 = 0.712^2 + 0.614^2 + 0.185 = 1.068$ and $\sigma_p = 1.033$. With $\mu_p = 0.055$, the VaR 1% is $-(0.055 - 2.326 \times 1.033) = 2.35\%$. The weighted sum of the two Normal VaRs is $0.5 \times 3.23 + 0.5 \times 2.82 = 3.03\%$, so the **diversification benefit** is 22% of the risk. With $\rho = 1$ it would be zero: the volatilities would simply add.

The Euler contributions $w_i(\Sigma w)_i/\sigma_p$, where $(\Sigma w)_i$ is the $i$-th component of the vector $\Sigma w$, add up exactly to $\sigma_p$, so they split the portfolio risk among the assets. The BET contributes 56% and the S&P 500 44%: the BET, with the higher volatility, carries more than half of the risk despite its equal weight.

## Correlation in Crisis Periods {h=3170c09c}

The chart shows the correlation of the daily BET and S&P 500 returns over rolling 250-day windows from 2001, with the two stress periods shaded and the whole-sample correlation, 0.21, as a dashed line. Before 2008 the rolling correlation was close to zero; it jumped in the autumn of 2008 and again, much higher, in March 2020.

Measured within each period, the correlation was 0.20 in 2017, a quiet year, 0.37 from September 2008 to March 2009 and 0.71 from February to May 2020. The rolling correlation reached its maximum, 0.67, on 25 March 2020, and stayed above 0.6 for a whole year: as long as the few weeks of March 2020 were inside the window, they dominated the correlation of 250 days. A Pearson correlation is driven by the largest co-movements, and crisis days are the largest.

The consequence for risk management is uncomfortable. Diversification is weakest exactly when it is needed: a portfolio VaR built on the average correlation of 0.21 is too optimistic in a crisis, when the correlation is between almost twice and more than three times the average. The variance–covariance VaR of the previous slide is therefore a calm-period number.

## Correlation and Tail Dependence {h=9abbeebc}

Pearson's correlation measures linear co-movement over all days, and ordinary days, which are the vast majority, dominate it [@EmbrechtsMcNeilStraumann2002]. It also depends on the marginal distributions and can change under monotone transformations of the returns. Rank correlations avoid this by using only the order of the returns: for the BET and the S&P 500, Kendall's $\hat\tau = 0.100$ and Spearman's rank correlation is 0.15.

Tail dependence asks directly about joint extremes. With $U_i = F_i(X_i)$ the returns transformed to a uniform scale by their distribution functions $F_i$, the lower tail dependence coefficient is

$$ \lambda_L = \lim_{q \to 0} P(U_1 \le q \mid U_2 \le q), $$ {#eq:taildep}

where $q$ is a small probability and $\lambda_L \in [0, 1]$. In words: the probability that the BET has one of its worst days, given that the S&P 500 has one of its worst days. Under independence the conditional probability equals $q$ and tends to zero.

In the data, the conditional probability is 0.22 at $q = 5\%$ and 0.19 at $q = 1\%$, far above the values 0.05 and 0.01 that independence would give. A joint crash, both indices below their own 1% quantile on the same day, occurs on 0.19% of the days, nineteen times the 0.01% expected under independence. A correlation of 0.21 does not reveal this.

## Copulas: an Introduction {h=2f813d79}

**Sklar's theorem** states that every joint distribution function can be written as

$$ F(x_1, x_2) = C(F_1(x_1), F_2(x_2)), $$ {#eq:sklar}

where $F_1$ and $F_2$ are the marginal distributions, each asset on its own, and $C$ is a **copula**, a joint distribution of uniform variables on $[0,1]^2$ [@McNeilFreyEmbrechts2015]. For continuous marginals the copula is unique. The theorem separates the two parts of a multivariate model: the copula contains all of the dependence, and the marginals contain all of the individual tails.

The separation gives a practical recipe. The marginals can be empirical or fitted (Student-t, EVT), whichever describes each asset best. The dependence is estimated on the **pseudo-observations** $\hat U_{i,t} = \mathrm{rank}(X_{i,t})/(n + 1)$, the ranks rescaled to the open interval $(0, 1)$, so that the copula estimate does not depend on the marginal model.

To compute a portfolio VaR, simulate pairs $(U_1, U_2)$ from the copula, map them back to returns with the quantile functions $F_i^{-1}$, compute the portfolio return and read VaR and ES from the simulated distribution. The textbook treatment is @FrankeHardleHafner2019, chapter 17.

## Gaussian and t Copulas {h=b0daabde}

The **Gaussian copula** is the dependence structure of a bivariate Normal distribution with correlation $\rho$. Its tail dependence is $\lambda_L = 0$ for every $\rho < 1$: in the limit, extreme days become independent, and joint crashes are rare. This is the copula implicit in the variance–covariance method.

The **t copula** is the dependence structure of a bivariate Student-t with correlation $\rho$ and $\nu$ degrees of freedom [@DemartaMcNeil2005]. Its tail dependence is positive:

$$ \lambda_L = 2\,t_{\nu+1}\Big(-\sqrt{(\nu + 1)(1 - \rho)/(1 + \rho)}\Big), $$ {#eq:tcopula}

where $t_{\nu+1}(\cdot)$ is the Student-t distribution function with $\nu + 1$ degrees of freedom. A smaller $\nu$ or a larger $\rho$ gives a larger $\lambda_L$: crashes occur together across assets even when the overall correlation is modest.

For the BET and the S&P 500, the correlation parameter of both copulas is estimated from Kendall's tau through $\rho = \sin(\pi\hat\tau/2) = 0.157$, a relation valid for all elliptical copulas, and the t copula has $\hat\nu = 4.50$ by maximum likelihood. The log-likelihood is 92.1 for the Gaussian copula and 219.5 for the t copula; the likelihood-ratio statistic, LR = 254.9, is far above the 5% critical value of $\chi^2(1)$, 3.84, so the t copula fits clearly better. Its implied tail dependence by @eq:tcopula is $\lambda_L = 0.096$, against 0 for the Gaussian copula.

## Joint Large Losses: Data and Two Copulas {h=f647bbcc}

The chart zooms into the lower-left corner of the uniform scale, below 0.25 on both axes: the pseudo-observations of the data (left) and samples of the same size, 6,466 points, from the fitted Gaussian copula (centre) and t copula (right). The square below 0.05 on both axes contains the days on which both indices were among their worst 5%.

The data have 70 such days, the Gaussian copula 37 and the t copula 55; under independence there would be about 16 ($6{,}466 \times 0.05 \times 0.05$). In the left panel the points concentrate visibly in the corner; in the centre panel they are spread almost as under independence; the right panel lies in between.

The Gaussian copula halves the number of joint bad days, even though it has the same rank correlation as the data. The t copula comes much closer, but still falls short of the data, a first sign that the dependence in the tail is stronger than a constant t copula can describe.

## Copula VaR: the Simulation Step by Step {h=596fb847}

The simulation has four steps. First, draw $(Z_1, Z_2)$ from a bivariate Normal with correlation $\rho$. For the Gaussian copula, set $U_i = \Phi(Z_i)$. For the t copula, also draw $W \sim \chi^2_\nu$ and set $U_i = t_\nu(Z_i\sqrt{\nu/W})$. The common factor $\sqrt{\nu/W}$ multiplies both coordinates; when $W$ is small, both become large in absolute value at once, which is why extremes occur together.

Second, map the uniforms back to returns with the empirical quantile functions, $R_i = \hat F_i^{-1}(U_i)$, so that each simulated return comes from the observed distribution of its index. Third, compute the portfolio return $R_p = 0.5R_1 + 0.5R_2$, and repeat the draw 200,000 times. Fourth, compute VaR 1% and ES 2.5% by historical simulation, @eq:hs, on the simulated values of $R_p$.

Both copulas use the same empirical marginals, so any difference between their VaRs comes only from the dependence. The design isolates the effect of tail dependence, which is the question of the section.

## Portfolio VaR 1% and ES 2.5% by Method {h=36e25266}

The table compares four estimates for the 50/50 portfolio over the 6,466 common days, in percent: VaR 1%, ES 2.5% and the probability of a joint crash at 1%. The variance–covariance method gives 2.35% and 2.36%; the Gaussian copula with empirical marginals 2.75% and 2.93%; the t copula with empirical marginals 2.81% and 3.03%; historical simulation on the portfolio returns 3.08% and 3.30%.

The ordering decomposes the error of the Normal method. It ignores both the heavy-tailed marginals and the tail dependence. Replacing the Normal marginals by the empirical ones, the step from the first to the second row, raises the VaR 1% by 0.40 percentage points; adding tail dependence, from the Gaussian to the t copula, raises it by a further 0.06 points and the joint-crash probability from 0.03% to 0.13%.

The t copula still underestimates the historical VaR, 2.81% against 3.08%, and the joint-crash probability, 0.13% against 0.19%. The BET–S&P 500 link is stronger in crises than a constant copula can describe: the dependence itself changes over time, as the rolling correlation showed. A copula whose parameters vary with time is the natural next step, and the open question of the [AI section](#sec-8).

## @end

Diversification between the BET and the S&P 500 removes 22% of the Normal VaR, but the correlation rises from 0.20 in a quiet year to 0.71 in the spring of 2020, and joint crashes are nineteen times more frequent than under independence. Copulas @eq:sklar separate the dependence from the marginals; the t copula fits the data far better than the Gaussian copula, but its portfolio VaR 1% of 2.81% remains below the historical 3.08%. The [next section](#sec-7) tests whether the VaR forecasts of the chapter were right.

# Backtesting

The section asks whether a VaR forecast was correct, using only the forecasts and the realised returns. It defines the hit sequence, derives the Kupiec test for the number of exceptions and the Christoffersen test for their clustering, presents the Basel traffic light, applies all three to the rolling forecasts of four indices from 2005 to 2026, ranks the forecasts with the quantile loss and ends with the Acerbi–Székely backtest of ES.

## The Hit Sequence {h=b5479988}

An **exception**, or exceedance, is a day on which the loss exceeded the VaR forecast, $r_t < -\mathrm{VaR}_t$. The hit sequence records them: $I_t = \mathbf{1}\{r_t < -\mathrm{VaR}_t\}$. The VaR must be the forecast made one day before, with no information from the future; a VaR estimated on a sample that includes day $t$ will be exceeded less often than a real forecast.

If the VaR at level $\alpha$ is correct, the hits $I_t$ are i.i.d. Bernoulli($\alpha$) [@Christoffersen1998]. This statement contains two separate properties. **Unconditional coverage**, $P(I_t = 1) = \alpha$, means that the number of exceptions is right on average. **Independence** means that an exception today does not make an exception tomorrow more likely: the timing of exceptions carries no information.

Under both properties, the number of exceptions in $n$ days, $x = \sum_t I_t$, follows a $\mathrm{Binomial}(n, \alpha)$ distribution, with mean $n\alpha$ and variance $n\alpha(1 - \alpha)$. Every test of the section compares the observed hit sequence with this benchmark.

## Exceptions in 250 Days under a Correct VaR 1% {h=6290ed45}

The chart shows the $\mathrm{Binomial}(250, 0.01)$ distribution: the number of exceptions in one year of trading for a VaR 1% that is exactly right. The bars are the probabilities, the labels the cumulative probabilities, and the colours the Basel zones: green for 0–4 exceptions, yellow for 5–9 and red for 10 or more.

A correct model has 2.5 exceptions on average, but the spread is large. It has no exception at all with probability 8.11%, and 5 or more with probability $100\% - 89.22\%$, about 11%. A correct model therefore lands in the yellow zone roughly one year in nine, while a model with no exceptions in a year is not necessarily too conservative.

One year of data separates a good model from a bad one only roughly. A model whose true exception rate is 2%, twice the nominal one, has 5 exceptions on average, and the binomial distribution around 5 overlaps heavily with the one around 2.5. The next slides quantify this low power.

## Kupiec's POF Test {h=47a28fd0}

The proportion-of-failures (POF) test of @Kupiec1995 tests unconditional coverage, $H_0$: $P(I_t = 1) = \alpha$. With $x$ exceptions in $n$ days and the observed rate $\hat\pi = x/n$, the likelihood-ratio statistic compares the binomial likelihood at the rate $\alpha$ with the likelihood at the rate $\hat\pi$:

$$ LR_{uc} = -2\big[(n - x)\ln(1 - \alpha) + x\ln\alpha - (n - x)\ln(1 - \hat\pi) - x\ln\hat\pi\big]. $$ {#eq:kupiec}

Under $H_0$, $LR_{uc}$ follows asymptotically a $\chi^2(1)$ distribution, as every likelihood-ratio test with one restriction ([Chapter 6](ch6.html)). It is never negative, because $\hat\pi$ maximises the likelihood, and the hypothesis is rejected at 5% when $LR_{uc} > 3.84$.

Step by step, with $n = 250$ and $x = 7$, so $\hat\pi = 0.028$: $\ln L_0 = 243\ln 0.99 + 7\ln 0.01 = -34.68$ and $\ln L_1 = 243\ln 0.972 + 7\ln 0.028 = -31.93$. Then $LR_{uc} = -2(-34.68 - (-31.93)) = 5.50$, with p-value 0.019: the model is rejected at 5%.

The test has low power with one year of data. In 250 days, any number of exceptions from 1 to 6 is accepted at 5%, so a model whose true rate is 2% passes the test with probability 76%. The test is two-sided: too few exceptions also lead to rejection, because an excessively prudent VaR wastes capital; with 250 days, however, only a year with no exception is rejected on that side.

## Kupiec on 21 Years of S&P 500 Forecasts {h=2b741fdb}

With a long sample, the same test becomes sharp. The GARCH-t VaR 1% of the S&P 500 from 3 January 2005 to 18 September 2026 covers $n = 5{,}460$ days and has $x = 95$ exceptions, an observed rate $\hat\pi = 0.0174$, against an expected $n\alpha = 55$. Under $H_0$, a 95% binomial interval for the observed rate runs from 0.75% to 1.26%; 1.74% is far outside it.

Step by step: $\ln L_0 = 5365\ln 0.99 + 95\ln 0.01 = -491.4$, $\ln L_1 = 5365\ln(1 - 0.0174) + 95\ln 0.0174 = -479.0$, and $LR_{uc} = -2(\ln L_0 - \ln L_1) = 24.73$, with a p-value below 0.001.

With 21 years of data, even a rate of 1.74% instead of 1% is rejected clearly; with one year it would have passed. The GARCH-t model tracks volatility well, as the crisis counts showed, but its exceptions are still too frequent. One probable cause is that the symmetric GARCH(1,1) ignores the leverage effect of [Chapter 9](ch9.html): volatility rises more after a fall than after a rise of the same size, so after bad days the forecast volatility is too low.

## Christoffersen's Independence Test {h=062f8726}

The Kupiec test counts exceptions but ignores when they happen. @Christoffersen1998 tests their timing by counting transitions: $n_{ij}$ is the number of days with $I_{t-1} = i$ and $I_t = j$. The rate of exceptions after a day without an exception is $\hat\pi_{01} = n_{01}/(n_{00} + n_{01})$, and after a day with an exception $\hat\pi_{11} = n_{11}/(n_{10} + n_{11})$. The null hypothesis $H_0$: $\pi_{01} = \pi_{11}$ says that an exception today does not change the chance of an exception tomorrow. The likelihood ratio compares a single rate $\hat\pi$ with the two transition rates:

$$ LR_{ind} = -2\ln\frac{(1 - \hat\pi)^{n_{00} + n_{10}}\hat\pi^{n_{01} + n_{11}}}{(1 - \hat\pi_{01})^{n_{00}}\hat\pi_{01}^{n_{01}}(1 - \hat\pi_{11})^{n_{10}}\hat\pi_{11}^{n_{11}}}, $$ {#eq:lrind}

asymptotically $\chi^2(1)$ under $H_0$. **Conditional coverage** combines both properties: $LR_{cc} = LR_{uc} + LR_{ind}$ is asymptotically $\chi^2(2)$, and a model is rejected at 5% if $LR_{cc} > 5.99$.

For the HS VaR 1% of the S&P 500 over 5,460 days, the transition counts are $n_{00} = 5313$, $n_{01} = 69$, $n_{10} = 69$ and $n_{11} = 8$. The exception rate after a quiet day is $\hat\pi_{01} = 1.28\%$, after an exception $\hat\pi_{11} = 10.4\%$: an exception is eight times more likely the day after another exception. The statistic $LR_{ind} = 19.43$ has a p-value below 0.001; the exceptions of HS cluster.

The test detects first-order dependence, exceptions on consecutive days. Clustering spread over a week, with quiet days in between, shows up only weakly in the transition counts, so the test is a minimal check of independence, not a complete one.

## The Basel Traffic Light {h=a005c896}

The Basel backtesting rule [@BCBS1996b] counts the exceptions of the one-day VaR 1% over the last 250 days. The table lists, for each count $x$, the zone, the cumulative probability $P(X \le x)$ under $\mathrm{Binomial}(250, 0.01)$, the plus factor and the resulting multiplier. In the green zone, 0–4 exceptions, nothing happens. In the yellow zone, 5–9 exceptions, the multiplier rises from 3 by a plus factor of 0.40 to 0.85. In the red zone, 10 or more exceptions, the multiplier is 4 and the model is presumed to be wrong.

The zones come from the binomial distribution of a correct model: the yellow zone starts where the cumulative probability exceeds 95% (95.88% at 5 exceptions), and the red zone where it reaches 99.99% (at 10 exceptions). A correct model thus falls in the yellow zone in about one year out of nine and in the red zone almost never.

The capital charge under the 1996 rules is the larger of yesterday's 10-day VaR 1% and $(3 + \text{plus factor})$ times the average 10-day VaR 1% over the last 60 days; in normal times the second term dominates. The FRTB keeps the count of VaR 1% exceptions, at the level of the bank and of each trading desk, under chapter MAR32 [@BCBS2023a].

## Exceptions over Time {h=4d349b68}

The chart shows the exceptions of the four rolling VaR 1% forecasts of the S&P 500 from 2005 to 2026; each mark is a day on which the method's VaR was exceeded, and the shaded bands are the two stress periods. For about 5,460 days, each method should have about 55 exceptions; the legend gives 77 for HS, 148 for the Normal method, 95 for GARCH-t and 86 for FHS.

The marks of HS and the Normal method form dense clusters: in 2007–2008, in 2011, in 2022, and for the Normal method also in 2018–2019 and 2025. Each cluster is a period in which volatility rose faster than the 500-day window could follow. Between the clusters there are long stretches with no exceptions at all, when the window still contained an earlier turbulent period.

The marks of GARCH-t and FHS are spread much more evenly over time, as independent exceptions should be. They are, however, more numerous in quiet years than those of HS: when volatility is low, the conditional VaR is low as well, and an ordinary bad day exceeds it. A correct conditional model would show exactly this even spread, but with about 55 marks in total.

## Backtest Results: S&P 500 and DAX {h=1cbf90ef}

The table summarises the backtests of the VaR 1% from 3 January 2005, about 5,500 days. For each method it gives the number of exceptions $x$, the rate in percent, the p-values of the Kupiec test (POF), of Christoffersen's independence test (ind.) and of the conditional coverage test (c.c.), the largest number of exceptions in any 250-day window, and the ES backtest statistic $Z_2$ of the last slide of this section with its p-value.

The Normal method is rejected by every test on both indices: 148 exceptions (2.71%) for the S&P 500 and 133 (2.41%) for the DAX, with up to 29 and 19 exceptions in a single 250-day window, deep in the red zone. HS has a nearly correct rate for the DAX, 1.21% with a Kupiec p-value of 0.121, but its exceptions cluster (independence p-value below 0.001), so it fails conditional coverage.

GARCH-t and FHS have independent exceptions, with independence p-values between 0.060 and 0.549, and a worst 250-day window of 9 or 10 exceptions instead of 12 to 29 for HS and the Normal method. Their rates, however, are 1.4% to 1.7%, so the Kupiec test rejects them on both indices. The conditional models solve the clustering problem, but not the level problem: their tails are still slightly too thin for the S&P 500 and the DAX.

## Backtest Results: BET and Bitcoin {h=668cb7fb}

The same tests for the BET from 2005 and Bitcoin from 1 January 2017 give a more favourable picture. For the BET, GARCH-t and FHS both have 65 exceptions, a rate of 1.20%, and pass all three tests: Kupiec p-value 0.158, independence 0.807 and 0.240, conditional coverage 0.358 and 0.185. HS has a similar rate, 1.22%, but its exceptions cluster (independence p-value 0.001), and the Normal method is rejected everywhere.

For Bitcoin, HS passes all three tests, with 33 exceptions, a rate of 0.93% and p-values 0.672, 0.316 and 0.553. FHS is now too prudent: 24 exceptions, a rate of 0.68%, rejected by the Kupiec test (p = 0.040) and by conditional coverage (p = 0.044). GARCH-t has 1.35% and is rejected by the Kupiec test at 5% (p = 0.045), though not for conditional coverage.

No method wins everywhere. The ranking depends on the market: the conditional methods are best for the equity indices, while Bitcoin's extreme and frequent volatility shifts are handled reasonably even by a two-year historical window. The Normal method is the only consistent result: it fails on every series.

## Ranking of the Forecasts and the Current Traffic-Light Zone {h=7eb46792}

Tests say whether a model is acceptable; a **scoring function** says which of two models is better. Since VaR is elicitable, the average quantile loss

$$ \bar S = \frac1n\sum_t(\alpha - I_t)(r_t + \mathrm{VaR}_t) $$ {#eq:avgscore}

ranks the forecasts [@Gneiting2011]: the lower the average, the better the forecasts. It rewards both the right exception rate and the tracking of volatility, since a forecast that is too high on quiet days is penalised by the term $\alpha(r_t + \mathrm{VaR}_t)$, and one that is too low on bad days by the term $(\alpha - 1)(r_t + \mathrm{VaR}_t)$.

For the equity indices, GARCH-t and FHS have the lowest average quantile loss (×100), for example 3.71 and 3.68 for the S&P 500 against 5.17 for HS and 5.80 for the Normal method, and the Normal method has the highest everywhere. For Bitcoin the four methods are close, from 13.53 (GARCH-t) to 14.52 (Normal), in line with the backtests of the previous slide.

The right half of the table counts exceptions in the last 250 days. All models are in the green zone today, except the Normal VaR of the BET, with 6 exceptions in the yellow zone. A recent window says little about a model: on the S&P 500, HS has 0 exceptions in the last year and GARCH-t 3, although over 21 years HS clusters its exceptions and GARCH-t does not.

## The Traffic Light through Time {h=ac114ed8}

The chart tracks the Basel count for the S&P 500: for every day from 28 December 2005, the end of the first 250-day window, the number of VaR 1% exceptions in the preceding 250 days, for HS, the Normal method and GARCH-t. The dashed lines are the upper limits of the green zone (4) and of the yellow zone (9); above the upper line is the red zone.

The Normal VaR was in the red zone on 33.3% of the days, HS on 11.8% and GARCH-t on only 0.5%; the largest counts were 29, 18 and 10. The Normal method reached 29 exceptions at the end of 2008, and it returned to the red zone in 2015–2016, 2019–2021, 2022–2023 and 2025. HS was in the red zone through most of 2008–2009 and again in 2020–2021.

A bank that had used the Normal VaR or HS would have paid the red-zone penalty for most of 2008–2009, exactly when capital was scarcest, while a bank using GARCH-t would have stayed in the yellow zone during the crisis, touching 10 exceptions only briefly. The traffic light rewards a model that tracks volatility, even one whose overall rate is too high.

## Backtesting ES: Acerbi and Székely {h=e763790c}

ES alone is not elicitable, but it can be tested together with VaR [@AcerbiSzekely2014]. Their second test statistic is

$$ Z_2 = \frac{1}{T\alpha}\sum_{t=1}^{T}\frac{r_t I_t}{\mathrm{ES}_t} + 1, \qquad I_t = \mathbf{1}\{r_t < -\mathrm{VaR}_t\}, \quad \alpha = 2.5\%, $$ {#eq:z2}

where $T$ is the number of test days and $\mathrm{VaR}_t$, $\mathrm{ES}_t$ the forecasts for day $t$. Each exception adds $r_t/\mathrm{ES}_t$, a negative number close to −1 if the loss is as large as the forecast ES. If VaR and ES are both correct, the expected sum is $-T\alpha$ and $E[Z_2] = 0$. A negative $Z_2$ means that the losses beyond the VaR are larger than the forecast ES, or that exceptions are too frequent, or both.

The distribution of $Z_2$ under the null depends on the forecast distributions, so its p-value is obtained by simulation: draw a return from each day's forecast distribution, recompute $Z_2$, repeat 1000 times, and compare the observed value with the simulated ones. The values and p-values are the last column of the two backtest tables.

For the S&P 500, the DAX and the BET, every method has a negative $Z_2$ with a p-value below 0.05: all four ES 2.5% forecasts are too small for the equity indices. FHS comes closest, with $Z_2$ between −0.18 and −0.20, because it takes the tail of $z$ from the data; the Normal method is furthest, between −0.83 and −1.15. For Bitcoin, HS (p = 0.224) and FHS ($Z_2 = 0.12$, p = 0.834) are not rejected.

## Case Study: VaR Models at Commercial Banks {h=d7b2a5b7}

@BerkowitzOBrien2002 were the first to study the VaR forecasts actually reported by large US banks, comparing them with the banks' daily trading P&L. The design is a backtest on real regulatory data: each bank's VaR 1% against its realised P&L, with a simple GARCH model of each bank's own P&L as a benchmark.

The banks' VaRs were conservative on average, with fewer exceptions than the nominal rate, but the exceptions that did occur came in clusters. The GARCH benchmark produced lower VaRs with comparable coverage: a simple model of the P&L series tracked changes in volatility better than the banks' structural models.

The same pattern appears in the data of this chapter: the window-based VaR is either too high in quiet periods or too slow in turbulent ones, while the conditional VaR adapts. A further line of work forecasts the quantile directly, without a model for the whole distribution: the CAViaR models of @EngleManganelli2004 specify an autoregression for the quantile itself and estimate it with the quantile loss @eq:pinball.

## @end

A correct VaR produces i.i.d. Bernoulli hits. The Kupiec test @eq:kupiec checks their rate, the Christoffersen test @eq:lrind their independence, the Basel traffic light turns the count into capital, and $Z_2$ @eq:z2 tests ES. Over 21 years, the Normal VaR fails everywhere, HS has the right rate on some series but clustered exceptions, and GARCH-t and FHS have independent exceptions but a rate of 1.4–1.7% on the S&P 500 and the DAX; all ES forecasts are too small for the equity indices. The [next section](#sec-8) turns one of the chapter's results into an open research question.

# AI for Scientific Discovery

The section starts from an open question raised by the portfolio results: whether the tail dependence between the Bucharest market and world markets jumps in crises and then returns to its previous level. It shows where AI tools can help a researcher who studies it and which checks they cannot replace.

## An Open Research Question {h=75b8180a}

Does the tail dependence between the BVB and world markets jump in crises and then revert? The chapter's results point in that direction. The BET–S&P 500 correlation was 0.20 in a quiet year and 0.71 in the spring of 2020; a constant t copula gives a tail dependence of $\lambda_L = 0.096$; and a portfolio VaR built on a constant copula stays below the historical VaR ([Portfolio VaR section](#sec-6)).

The question remains open for three reasons. Crises are few, so any estimate of crisis dependence rests on a handful of episodes. The BVB trades at different hours from New York, so the same-day dependence mixes true co-movement with timing effects. And tail estimates need many observations, while a crisis window contains only a few joint bad days.

Related work uses entropy as a measure of market risk, a different route to the tail: @PeleLazarDufour2017 and @PeleMazurencuMarinescuPele2019 are further reading. AI tools can speed up a study of this question, but their output must be checked [@WangEtAl2023].

## Possible Uses of AI Tools {h=ab19ac2f}

An AI assistant can help at three stages. For the literature, it can list studies of time-varying copulas and of contagion between emerging and developed markets, a starting point for placing the BET in context. For the code, it can write a first version of rolling-window t-copula estimates for the BET and the S&P 500 over 250-day windows, with $\rho$, $\nu$ and $\lambda_L$ plotted over time. For robustness, it can propose and implement variants: two-day returns to absorb the different trading hours, the DAX instead of the S&P 500, other window lengths.

The example prompt on the slide is precise: it names the copula, the estimation method, the data, the window length of 250 days, the step of 21 days and the three outputs. A precise prompt produces code whose choices can be checked against the definitions of this chapter; a vague one produces code whose choices remain hidden.

The output of each stage is a draft. The list of papers has to be checked reference by reference, and the code has to be tested on known results, for example by reproducing the full-sample t-copula estimates, $\rho = 0.157$ and $\hat\nu = 4.50$, before it is used on rolling windows.

## What to Check {h=b9f51683}

Several errors are common in AI-generated risk code. The first is the VaR convention: "VaR 1%" is the loss exceeded with probability 1%, a positive number; an answer that uses the complementary probability, or reports a negative VaR, mixes conventions. The second is data alignment: prices must be aligned on common trading days before returns are computed, and the BVB and New York trade at different hours.

The third concerns the copula itself. Pseudo-observations are the ranks divided by $n + 1$, not by $n$: otherwise the largest observation sits exactly at 1, where the quantile transforms $\Phi^{-1}$ and $t_\nu^{-1}$ are infinite and the copula log-likelihood is undefined. Tail dependence estimated from 250 days rests on a few joint bad days, so its uncertainty must be reported, for example with the bootstrap of [Chapter 2](ch2.html).

Overlapping windows share most of their data, so consecutive estimates are not independent evidence: a rolling series that crosses a threshold several times is one finding, not several. Finally, every reference must exist: the DOI of each cited paper has to be checked, because invented references with plausible titles are a known failure of language models.

## Project Idea {h=50298d17}

The project asks whether a time-varying copula gives a better portfolio VaR than a constant copula for a portfolio of BVB and world equities. The data are the daily closes of the BET, the BET-TR (the BET total return index, with dividends reinvested), the S&P 500 and the DAX from 2005, from EODHD (EOD Historical Data).

The steps follow the chapter. First, fit GARCH-t marginals to each index ([Chapter 9](ch9.html)) and model the dependence with a constant t copula and with a t copula estimated on a rolling window. Second, compute out-of-sample one-day VaR 1% and ES 2.5% for a 50/50 portfolio from 2008. Third, apply the Kupiec, Christoffersen and $Z_2$ tests to both variants, with special attention to 2008, 2020 and 2022.

The deliverable is a table, a chart and a paragraph on what the data can and cannot show. Any use of AI tools is declared, together with the AI errors found and corrected.

## @end

The BET–S&P 500 correlation rises from 0.20 in a quiet year to 0.71 in a crisis, and a constant copula underestimates the portfolio VaR. Whether tail dependence jumps and reverts is open, because crises are few and trading hours differ. AI tools can draft the literature search and the rolling-copula code; the researcher checks the VaR convention, the alignment, the pseudo-observations, the uncertainty and the references.

# Summary

The chapter's results reduce to a few statements about measuring, estimating and testing tail risk, a set of formulas and three self-check questions.

## Key Takeaways {h=537507b2}

VaR 1% is the negative of the 1% quantile, @eq:var: the loss exceeded on one day in a hundred. ES 2.5%, @eq:es, is the average loss on the worst 2.5% of the days. ES is coherent; VaR can penalise diversification, as in the two-bond example. VaR is elicitable, while ES is elicitable only jointly with VaR.

On daily data, the Normal VaR 1% is 14–24% too small on all six series. Historical simulation, the Student-t and EVT agree at 1%, and Cornish–Fisher breaks down when the kurtosis is large. Conditional VaR, GARCH-t and FHS, tracks turbulent periods and passes tests that the window-based methods fail, above all the independence of exceptions.

For portfolios, correlation rises in crises, and tail dependence, captured by the t copula, matters for joint crashes. Backtesting combines four tools: the Kupiec test for the rate of exceptions, the Christoffersen test for their clustering, the Basel traffic light for regulation and the $Z_2$ statistic for ES.

## Key Formulas {h=1b35c24c}

The table collects the chapter's formulas. VaR and ES are defined by @eq:var and @eq:es, and estimated by historical simulation with @eq:hs. For the Normal distribution they have the closed forms @eq:normal, for the Student-t @eq:tdist, and the EVT estimator @eq:pot gives both from a GPD fitted above a threshold.

The conditional VaR @eq:cond scales the quantile of the standardised innovation by tomorrow's volatility. The portfolio variance @eq:portvar combines volatilities and correlations, and the tail dependence of the t copula is @eq:tcopula.

The Kupiec statistic @eq:kupiec tests the exception rate, the Christoffersen statistic @eq:lrind their independence, and their sum $LR_{cc}$ tests conditional coverage against $\chi^2(2)$. The Acerbi–Székely statistic @eq:z2 tests ES and VaR together.

## Self-Assessment {h=8add9cc9}

The first question gives a division's report "VaR 1% = −2.4%". The sign is wrong: −2.4% is the quantile $q_{0.01}$ of the returns, and by @eq:var the VaR 1% is $-q_{0.01} = 2.4\%$, a positive loss.

The second question describes a model with 3 exceptions in 250 days, all in the same week. The Kupiec test does not detect the problem, because 3 is close to the expected 2.5. The Christoffersen independence test can detect it if the exceptions fall on consecutive days, because these produce transitions from an exception to an exception; the test looks only at day-to-day transitions, so three exceptions on Monday, Wednesday and Friday would leave it blind; and with only three exceptions in a year its power is limited, which is why the test is applied to long samples.

The third question asks whether two portfolios with the same VaR 1% can have different ES 2.5%. They can: ES depends on the whole tail beyond the quantile, VaR on a single point. The next chapter, [Chapter 11](ch11.html), turns to the fractal market hypothesis and long memory.
