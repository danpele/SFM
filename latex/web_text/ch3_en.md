<!--
Course text of the web version, Chapter 3 (EN). Inserted into the page by latex/build_web_lecture.py.
Format: # section title (introduction), ## slide title {h=hash} (text), @after (text after the live block), ## @end (summary).
Numbered equations: $$ ... $$ {#eq:label}, reference @eq:label. Citations: @Key -> Author (year), [@Key] -> (Author, year); keys come from the deck's bibliography.
Current hashes: python3 build_web_lecture.py 3 --hashes
-->

## @intro

@text
Chapter 2 showed that daily returns have heavier tails than the Normal distribution. A model for daily returns, however, is never only a model for one day. Log returns add up over time, so a monthly log return is the sum of about 21 daily log returns, and the law of the daily return fixes the law of every longer horizon. The chapter asks which probability laws can describe daily returns and, at the same time, their weekly and monthly sums.

The answer of probability theory is the family of stable laws. They are the only laws whose sums keep their shape, and, by the generalised central limit theorem (GCLT), the only possible limits of normalised sums of independent and identically distributed (i.i.d.) variables. @Mandelbrot1963 proposed them for cotton prices, with an index of stability $\alpha \approx 1.7$, which implies an infinite variance. The chapter develops the theory of these laws, their simulation and their estimation, and then fits them to daily log returns of four series: the BET index of the Bucharest Stock Exchange, the S&P 500, the DAX and Bitcoin.

On these series the maximum likelihood (ML) estimates of $\alpha$ lie between 1.410 and 1.606, far from the Normal value 2, and the stable law fits the centre of the distribution much better than the Normal distribution. In the far tails, however, it predicts many more extreme losses than were observed: for the S&P 500, 19.1 days with a loss above 10% in 6,714 days, against one observed day. The last part of the chapter tests the infinite-variance hypothesis directly and concludes that daily returns have heavy tails but a finite variance.

@chart sfm_ch3_fit_density_1
Daily log returns of the BET and the S&P 500 (dots, histogram on a log scale) with three fitted laws: Normal (red), Student-t (green) and stable, fitted by maximum likelihood (blue); the horizontal axis is the daily log return in %, the vertical axis the density on a log scale.

@goals
- **State** the stability property and compute the scale of a sum of stable variables.
- **Explain** the generalised central limit theorem and when sums do not approach the Normal distribution.
- **Interpret** the four parameters $(\alpha, \beta, \gamma, \delta)$ and convert between the S0 and S1 parameterisations.
- **Determine** which moments exist for a given $\alpha$ and compare tail probabilities with those of the Normal distribution.
- **Simulate** stable variables and estimate their parameters by quantiles and by maximum likelihood.
- **Evaluate**, on BET, S&P 500, DAX and Bitcoin returns, where the stable model works and where it fails.

@plan
- Mandelbrot and the Cotton Prices | The historical origin: cotton prices, Mandelbrot's hypothesis, Fama's tests and the question of the law of a sum.
- Stability under Summation | The definition of stable laws, the scale rule $c^\alpha = a^\alpha + b^\alpha$ and its consequences for horizons and diversification.
- The Generalised Central Limit Theorem | Stable laws as the only limits of normalised sums; domains of attraction.
- Parameters and the Characteristic Function | The four parameters, the S0 and S1 parameterisations and the three laws with a closed-form density.
- Tails and Moments | Power-law tails, the existence of moments and the consequences of an infinite variance for risk measures.
- Simulation | The Chambers–Mallows–Stuck method, in the symmetric and in the skewed case.
- Estimation | McCulloch's quantile method and maximum likelihood, compared in a Monte Carlo study.
- Fitting Real Returns | Stable, Student-t and Normal laws fitted to the BET, S&P 500, DAX and Bitcoin.
- Critique of the Stable Model: Finite or Infinite Variance | Three tests of the infinite-variance hypothesis and the explanations of the apparent stable shape.
- AI for Scientific Discovery | An open question on the tails of Bitcoin, the checks it needs and a project idea.
- Summary | The formulas of the chapter and questions for self-assessment.

# Mandelbrot and the Cotton Prices

The stable laws entered finance through an empirical observation, not through theory. This section tells how Benoit Mandelbrot came to propose them, how Eugene Fama tested the proposal on stock returns, and why the question is really about the law of a sum of returns.

## Cotton: an Old Market with Long Price Records {h=fdb55007}

In the nineteenth century cotton was one of the most traded commodities in the world. The exchanges of New Orleans, New York and Liverpool quoted prices every day, and these quotes produced some of the longest daily and monthly price series available in economics. Long series matter for the study of tails: extreme moves are rare by definition, and only a long record contains enough of them to say how often they occur.

In the early 1960s these series reached Benoit Mandelbrot, then a researcher at IBM (International Business Machines). He did not study the price level but the changes of the logarithm of the price, that is, the log returns of Chapter 1. His paper, @Mandelbrot1963, is the starting point of the study of heavy tails in finance: it asked which probability law describes these changes, and it gave an answer that departed from the Normal distribution used until then.

The painting on the slide, Edgar Degas' *A Cotton Office in New Orleans* (1873), shows the trade that generated these records.

## Mandelbrot's Observations on Cotton Prices {h=14deb163}

Mandelbrot's first observation was that the cotton series contained far more large changes than the Normal distribution allows. A related symptom was that the sample variance did not settle as the sample grew: each new extreme change pushed it up again, instead of letting it converge to a fixed value as it would for a variable with a finite variance.

The second observation was that the shape of the distribution was the same at every horizon. After rescaling, daily and monthly changes looked alike, which means that the sum of many daily changes kept the shape of a single daily change. Under the Normal distribution this property is familiar, since a sum of Normal variables is again Normal. Mandelbrot saw it together with heavy tails, which the Normal distribution cannot produce.

His proposal combined the two observations: price changes follow stable Paretian laws with an index $\alpha \approx 1.7$ [@Mandelbrot1963]. "Paretian" refers to tails that decay like a power of $x$, as in Pareto's law of incomes. The consequence, discussed throughout this chapter, is that the variance of price changes would be infinite.

## Fama's Tests on Stock Returns {h=6e53a6bb}

@Fama1965 took the hypothesis from commodities to stocks. He studied the daily returns of the 30 stocks of the Dow Jones Industrial Average and found, for every stock, more large returns than the Normal distribution predicts. His estimates of the index $\alpha$ were below 2, which supported Mandelbrot's hypothesis.

Estimation methods for symmetric stable laws followed. @FamaRoll1968 and @FamaRoll1971 derived estimators based on sample quantiles, together with the tables needed to use them. These methods are the ancestors of the quantile estimator of McCulloch presented in the [estimation section](#sec-7).

Within less than ten years the evidence started to turn against the infinite variance. The [critique section](#sec-9) at the end of the chapter returns to these studies and repeats their tests on modern data.

## The Importance of the Law of a Sum {h=d82ab6de}

A monthly log return is the sum of about 21 daily log returns. A model for daily returns therefore implies a model for weekly, monthly and yearly returns, and a good model must remain consistent across horizons. The Normal distribution passes this test, since it keeps its shape under summation (Chapter 2), but it has thin tails and misses the large daily moves.

A natural candidate with heavy tails is the Student-t distribution. Question for the room: is the sum of 21 independent Student-t variables again a Student-t variable? The answer is no. The Student-t family is not closed under summation, and if its variance is finite, the central limit theorem (CLT) pushes the sum towards the Normal distribution. A Student-t model for daily returns therefore implies an approximately Normal model for monthly returns.

The goal of the next sections follows from this answer: to find every law whose sums keep the same shape. These are the stable laws.

## @end

Mandelbrot's two observations, too many large changes and the same shape at every horizon, led to the hypothesis of stable Paretian laws with $\alpha \approx 1.7$ and an infinite variance. Fama's study of the Dow Jones stocks supported it, and the question it raises is about the law of a sum of returns. The [next section](#sec-2) defines the laws that keep their shape under summation.

# Stability under Summation

This section defines stable laws through their behaviour under summation and derives the rule that gives the scale of a sum. The rule depends on a single number, the index of stability $\alpha$, and has direct consequences for the risk of long horizons and for diversification.

## Definition: Stable Laws {h=0d68e849}

A random variable $X$ is stable if, for two independent copies $X_1, X_2$ of $X$ and any weights $a, b > 0$, there exist a scale $c > 0$ and a shift $d$ such that

$$ aX_1 + bX_2 \overset{d}{=} cX + d, $$ {#eq:stable}

where $\overset{d}{=}$ means "has the same distribution as". In words, a weighted sum of two copies has the same shape as one copy; only the scale $c$ and the location $d$ change. If $d = 0$ for all $a$ and $b$, the law is strictly stable.

By induction, the property extends to $n$ independent copies: $X_1 + \dots + X_n \overset{d}{=} c_n X + d_n$. The scale factor cannot be arbitrary. The only possible factor is $c_n = n^{1/\alpha}$, with $0 < \alpha \le 2$ [@Nolan2020]. The exponent $\alpha$ is called the index of stability, also the characteristic exponent or the tail index, and it indexes the whole family.

The upper bound $\alpha \le 2$ is not a convention. For $\alpha = 2$ the factor is $\sqrt{n}$, the familiar rule for sums of variables with a finite variance; no stable law has a factor growing more slowly than $\sqrt{n}$.

## Two Examples: Normal and Cauchy {h=d831fcd7}

The Normal distribution is the first example. If $X_1, X_2 \sim N(0, \sigma^2)$ are independent, then $aX_1 + bX_2 \sim N(0, (a^2 + b^2)\sigma^2)$, which has the shape of $X$ with scale $c = \sqrt{a^2 + b^2}$. Variances add, so $c^2 = a^2 + b^2$: this is the case $\alpha = 2$.

The second example is the Cauchy distribution, with density $f(x) = 1/(\pi(1 + x^2))$. A weighted sum $aX_1 + bX_2$ of two independent standard Cauchy variables is Cauchy with scale $c = a + b$. Scales add linearly, $c^1 = a^1 + b^1$: this is the case $\alpha = 1$.

The Cauchy case has a striking consequence. Taking $a = b = 1/n$ for $n$ copies, the average of $n$ Cauchy draws is again standard Cauchy: averaging does not reduce the dispersion at all. The law of large numbers fails because the Cauchy distribution has no mean.

## The General Rule and a Worked Example {h=9aabaea5}

The two examples are special cases of one rule. For independent copies $X_1, X_2$ of a strictly stable $X$ with index $\alpha$, the scale of the weighted sum satisfies

$$ c^\alpha = a^\alpha + b^\alpha, \qquad c = (a^\alpha + b^\alpha)^{1/\alpha}. $$ {#eq:scale}

The worked example takes $a = 3$ and $b = 4$. For $\alpha = 2$ the scale of the sum is $c = \sqrt{9 + 16} = 5.00$; for $\alpha = 1.5$ it is $c = (3^{1.5} + 4^{1.5})^{1/1.5} = 5.58$; for $\alpha = 1$ it is $c = 3 + 4 = 7.00$. The smaller $\alpha$, the larger the scale of the sum. With heavy tails a single large term can dominate the whole sum, so the sum is wider than the Normal rule suggests.

What do you think? For $\alpha = 1.5$, what is the scale factor of a sum of two copies? Setting $a = b = 1$ in @eq:scale gives $c_2 = 2^{1/1.5} = 2^{2/3} = 1.59$, larger than the factor $\sqrt{2} \approx 1.41$ of the Normal distribution.

## Stability in a Simulation {h=5d42d7d8}

The chart checks the stability property by simulation. It uses the notation of the [parameters section](#sec-4): $S(\alpha, \beta, \gamma, \delta)$ is the stable law with index $\alpha$, skewness $\beta$, scale $\gamma$ and location $\delta$. Each point is a pair of quantiles: on the horizontal axis a quantile of one draw of $S(1.7, 0, 1, 0)$, on the vertical axis the same quantile of a rescaled sum of 10 draws. If the two laws are equal, the points lie on the 45-degree line.

The blue points divide the sum of 10 draws by $10^{1/1.7} = 3.87$, the factor $n^{1/\alpha}$ of stability. They lie on the 45-degree line over the whole range: the rescaled sum has the same law as one draw. The red points divide the same sums by $\sqrt{10} = 3.16$, the rule of the Normal distribution. The division is too weak, so the tails of the rescaled sum remain too wide: its 99% quantile is 6.31, against 5.04 for a single draw.

The lesson for returns is that the square-root rule used to scale volatility from one horizon to another is a property of $\alpha = 2$. Under a stable law with $\alpha < 2$ it understates the spread of the sum.

## Consequences: Horizons and Diversification {h=aa8a4eb9}

The first consequence concerns the horizon. If daily returns are i.i.d. $S(\alpha, 0, \gamma, 0)$, the sum over 20 days has scale $20^{1/\alpha}\gamma$. For $\alpha = 2$ this is $4.47\,\gamma$, the square-root-of-time rule; for $\alpha = 1.7$ it is $5.83\,\gamma$; for $\alpha = 1.5$ it is $7.37\,\gamma$. With $\alpha < 2$, the risk of long horizons grows faster than $\sqrt{n}$.

The second consequence concerns diversification. An equally weighted portfolio of $N$ i.i.d. stable assets is $\frac{1}{N}(X_1 + \dots + X_N)$, whose scale is $N^{1/\alpha}\gamma/N = N^{1/\alpha - 1}\gamma$. For $N = 10$ assets, $\alpha = 2$ gives $0.32\,\gamma$, $\alpha = 1.5$ gives $0.46\,\gamma$, and $\alpha = 1$ gives $1.00\,\gamma$.

With $\alpha = 1$, diversification does not reduce the scale at all: the portfolio is as dispersed as a single asset. @Fama1965 discussed this consequence of the stable hypothesis for portfolio choice. For $1 < \alpha < 2$ diversification still helps, but less than under the Normal distribution.

## Recap: Stability {h=a420f589}

A stable law is one whose sums of independent copies keep the same shape, only rescaled and shifted. The scale of a weighted sum follows @eq:scale, and the sum of $n$ copies has the factor $n^{1/\alpha}$ with $0 < \alpha \le 2$. The Normal distribution is the case $\alpha = 2$ and the Cauchy distribution the case $\alpha = 1$.

The smaller $\alpha$, the faster risk grows with the horizon and the less diversification helps. Both effects matter for returns only if $\alpha < 2$ really describes them, which the last sections of the chapter test.

## @end

Stability under summation singles out a one-parameter rule for scales: $c^\alpha = a^\alpha + b^\alpha$, and $n^{1/\alpha}$ for $n$ copies. The simulation confirmed that a sum of 10 draws with $\alpha = 1.7$, divided by $10^{1/1.7}$, has the law of one draw, while the Normal factor $\sqrt{10}$ leaves its tails too wide. The [next section](#sec-3) shows why these laws are not only convenient but unavoidable as limits of sums.

# The Generalised Central Limit Theorem

The classical central limit theorem explains why the Normal distribution appears whenever many small independent effects are added. This section states its generalisation, which covers variables with an infinite variance, and shows that the tail of a single return decides the law of long sums.

## From the Classical to the Generalised CLT {h=ad424e6b}

The classical central limit theorem (CLT, Chapter 2) concerns i.i.d. variables $X_i$ with mean $\mu$ and finite variance $\sigma^2$. It states that

$$ \frac{X_1 + \dots + X_n - n\mu}{\sigma\sqrt{n}} \to N(0, 1) $$

in distribution. If the variance is infinite, this statement has no content: $\sigma$ does not exist, and normalising by $\sqrt{n}$ is the wrong rate.

The generalised central limit theorem [@GnedenkoKolmogorov1954] answers the question without assuming a finite variance. If, for some constants $a_n > 0$ and $b_n$ that rescale and recentre the sum, the normalised sum $(X_1 + \dots + X_n - b_n)/a_n$ converges to a non-degenerate law, then that law is stable. For the classical CLT the constants are $a_n = \sigma\sqrt{n}$ and $b_n = n\mu$.

Stable laws are therefore the only possible limits of normalised sums of i.i.d. variables, and the Normal distribution is the special case $\alpha = 2$. This is the deeper reason to study them: any model in which returns are sums of many independent shocks leads, in the limit, to a stable law.

## Domains of Attraction {h=259d0c0a}

The domain of attraction of a stable law is the set of all laws whose normalised sums converge to it. If the variance is finite, the law belongs to the domain of attraction of the Normal distribution, and the normalising constant grows like $\sqrt{n}$. If the tails follow a power law, $P(|X| > x) \approx C x^{-\alpha}$ with a constant $C > 0$ and $0 < \alpha < 2$, the law belongs to the domain of attraction of the stable law with the same $\alpha$, and $a_n \propto n^{1/\alpha}$, where $\propto$ means proportional to.

The consequence is that the tail of the distribution of one return decides the law of sums with many terms. The centre of the distribution does not matter in the limit; only the rate at which large values become rare does. A law with a power tail of index below 2 leads to a non-Normal stable limit, whatever its shape near zero.

Paul Lévy characterised the stable laws in the 1920s, and the complete theory of limit laws for sums is in @GnedenkoKolmogorov1954.

## The GCLT in a Simulation {h=37ee5f47}

The chart illustrates the GCLT with a law that is not stable: the Student-t distribution with $\nu = 1.5$ degrees of freedom. It has infinite variance and tails proportional to $x^{-1.5}$, so its domain of attraction is that of a stable law with $\alpha = 1.5$. Sums of $n$ draws are divided by $n^{1/1.5}$ and their histograms are drawn on a log scale, for $n = 1$, 10 and 100.

As $n$ grows, the histograms approach the stable limit $S(1.5, 0, 1.53, 0)$, drawn as the solid dark line. The tail probabilities converge as well: the probability that the normalised sum exceeds 10 in absolute value is 2.73% for $n = 100$ and 2.64% under the limit law. The histogram for $n = 1$ is visibly more peaked in the centre than the limit, but its tails already have the limiting slope.

The dashed red curve is a Normal distribution with the same interquartile range as the limit. Its probability of an absolute value above 10 is 0.0005%: on the log scale it drops below the histograms long before the tails end. A Normal approximation, fitted to the centre, misses the tails completely.

## What the GCLT Means for Returns {h=420f808a}

The GCLT divides the possible models of daily returns into two worlds. If the tail index of daily returns is below 2, the variance is infinite and monthly returns are stable with the same $\alpha$ as daily returns. If the tail index is above 2, the variance is finite and monthly returns move towards the Normal distribution.

The two worlds give different, testable predictions about returns at longer horizons: in the first, the estimated $\alpha$ should stay the same as the horizon grows; in the second, it should rise towards 2. Question for the room: in which world do the S&P 500 returns live?

The answer requires data. The [critique section](#sec-9) estimates $\alpha$ for daily, weekly and monthly returns of the four series of the chapter and compares the two predictions.

## Recap: The Generalised CLT {h=1613487c}

The classical CLT requires a finite variance, uses the normalisation $\sqrt{n}$ and gives a Normal limit. The GCLT drops the variance assumption and shows that stable laws are the only possible limits of normalised sums.

Laws with power-law tails of index $\alpha < 2$ need the normalisation $n^{1/\alpha}$, and their sums converge to the stable law with the same $\alpha$. The tail of a single return therefore decides the law of long sums.

## @end

The GCLT makes the stable laws the natural limits of sums of i.i.d. shocks, with the Normal distribution as the only finite-variance case. Which limit applies depends only on the tail of one return: a tail index below 2 leads to a stable limit with the same $\alpha$, a tail index above 2 to the Normal distribution. The two cases give testable predictions about the horizon behaviour of $\hat\alpha$, which the [critique section](#sec-9) checks. To work with stable laws we first need a way to write them down, the subject of the [next section](#sec-4).

# Parameters and the Characteristic Function

Apart from three cases, stable laws have no closed-form density. They are defined instead by their characteristic function, which has a simple form and makes the stability property visible. This section introduces the four parameters, the two parameterisations used in practice and the traps they create in software.

## The Characteristic Function {h=a2931b14}

The characteristic function of a random variable $X$ is $\varphi_X(t) = E[e^{itX}] = E[\cos(tX)] + i\,E[\sin(tX)]$, for real $t$, where $i = \sqrt{-1}$ is the imaginary unit and $t$ is a real argument, not time. It always exists, because $|e^{itX}| = 1$, even when no moment of $X$ exists, and it determines the distribution uniquely. Both properties make it the natural tool for laws with infinite variance.

Two properties are used throughout the chapter. For independent $X$ and $Y$, $\varphi_{X+Y}(t) = \varphi_X(t)\,\varphi_Y(t)$: sums of independent variables become products of characteristic functions. For constants $a$ and $b$, $\varphi_{aX+b}(t) = e^{ibt}\varphi_X(at)$, so rescaling and shifting have a simple effect. For the Normal distribution $N(\mu, \sigma^2)$, $\varphi(t) = \exp(i\mu t - \sigma^2 t^2/2)$.

The reason to work with $\varphi$ is practical: apart from three cases, stable laws have no density in closed form. They are defined by their characteristic function, and densities, distribution functions and quantiles are computed numerically from it.

## The Stable Characteristic Function (S1) {h=d1680b7d}

In the S1 parameterisation [@SamorodnitskyTaqqu1994; @Nolan2020], $X \sim S(\alpha, \beta, \gamma, \delta; 1)$ if, for $\alpha \ne 1$,

$$ \varphi(t) = \exp\Big\{-\gamma^\alpha |t|^\alpha \big[1 - i\beta\,\mathrm{sign}(t)\tan\tfrac{\pi\alpha}{2}\big] + i\delta t\Big\}, $$ {#eq:cf-s1}

and, for $\alpha = 1$, $\varphi(t) = \exp\{-\gamma |t| [1 + i\beta\,\frac{2}{\pi}\mathrm{sign}(t)\ln|t|] + i\delta t\}$. Here $\mathrm{sign}(t)$ is $+1$ for $t > 0$ and $-1$ for $t < 0$; the term with $\beta$ is the imaginary part that makes the law asymmetric.

The four parameters have distinct roles. The index of stability $\alpha \in (0, 2]$ controls the tails: the smaller $\alpha$, the heavier the tails. The skewness $\beta \in [-1, 1]$ controls asymmetry: $\beta > 0$ gives a heavier right tail and $\beta < 0$ a heavier left tail. The scale $\gamma > 0$ plays the role of the standard deviation $\sigma$, and the location $\delta$ is a real shift.

Stability can be read directly from @eq:cf-s1. The sum of $n$ independent copies has characteristic function $\varphi(t)^n$, which has the same form with $\gamma^\alpha$ replaced by $n\gamma^\alpha$ (and $\delta$ by $n\delta$). The scale of the sum is therefore $n^{1/\alpha}\gamma$, as in the [stability section](#sec-2).

## The Effect of α {h=d5d23aa0}

The chart shows symmetric stable densities $S(\alpha, 0, 1, 0)$ for $\alpha$ from 0.7 to 2, on a linear scale (left) and on a log scale (right). On the linear scale, a smaller $\alpha$ gives a taller and narrower peak; the mass removed from the shoulders moves partly to the centre and partly to the tails.

The log scale shows the tails, which the linear scale hides. The Normal density ($\alpha = 2$, here with variance 2) falls like $e^{-x^2/4}$, a downward parabola on the log scale. All the other curves fall like a power of $x$, which on this scale looks like a slow, almost flat decline. At $|x| = 8$ the Normal density is already several orders of magnitude below the stable ones.

## The Effect of β and γ {h=ccb1f8b9}

The left panel varies the skewness $\beta$ for $\alpha = 1.3$ and $\gamma = 1$. The parameter moves weight between the two tails: $\beta = 1$ (green) has the heavier right tail and $\beta = -1$ (red) the heavier left tail, while the centre of the density changes little. For returns, a negative $\beta$ is the stable description of crash risk, a left tail heavier than the right one.

The right panel varies the scale $\gamma$ for $\alpha = 1.7$ and $\beta = 0$. The scale stretches the density: doubling $\gamma$ doubles every quantile around $\delta$ and halves the height of the peak. In this respect $\gamma$ behaves like a standard deviation, but it remains well defined when the variance is infinite.

## Two Parameterisations: S0 and S1 {h=82725c3b}

The S0 parameterisation, recommended by Nolan for data analysis, uses for $\alpha \ne 1$ the characteristic function

$$ \varphi(t) = \exp\Big\{-\gamma^\alpha |t|^\alpha \big[1 + i\beta\,\mathrm{sign}(t)\tan\tfrac{\pi\alpha}{2}\big(|\gamma t|^{1-\alpha} - 1\big)\big] + i\delta_0 t\Big\}. $$

Only the location differs between the two parameterisations. The parameters $\alpha$, $\beta$ and $\gamma$ are the same in both, and the locations $\delta_0$ (in S0) and $\delta_1$ (in S1) are related by

$$ \delta_0 = \delta_1 + \beta\gamma\tan\tfrac{\pi\alpha}{2}, \qquad \alpha \ne 1. $$ {#eq:s0s1}

For $\beta = 0$ the two coincide.

Each parameterisation has its use. S1 gives simpler algebra for sums and is used in most theoretical books [@SamorodnitskyTaqqu1994]. In S0 the density changes continuously with $\alpha$ and $\beta$, which makes it better for estimation and for plots [@Nolan2020]. The [next frame](#s25) shows why continuity is the decisive property.

## S1 vs S0 near α = 1 {h=63a9e908}

The chart draws four laws with $\beta = 0.8$, $\gamma = 1$ and location 0, for $\alpha = 0.8$, 0.95, 1.05 and 1.2, first in S1 (left) and then in S0 (right). In S1 the density runs away to $+\infty$ as $\alpha \uparrow 1$ and comes back from $-\infty$ as $\alpha \downarrow 1$. The reason is the term $\tan\frac{\pi\alpha}{2}$ in @eq:s0s1, which tends to infinity as $\alpha$ approaches 1: a small change of $\alpha$ moves the whole law far along the axis.

In S0 the same four laws are shifted so that the mode stays near 0 and changes smoothly with $\alpha$. For estimation this is the decisive property: an optimiser that moves $\alpha$ slightly sees a slightly different density, not a law that has jumped across the axis. This is why S0 is used for fitting and S1 for algebra.

## Parameterisations of Stable Laws in Software {h=d6329d87}

The parameterisation is a frequent source of errors in software. The function scipy.stats.levy_stable uses S1 by default; it is switched to S0 with levy_stable.parameterization = 'S0', and its arguments are levy_stable.pdf(x, alpha, beta, loc=delta, scale=gamma). Papers and programs report $\delta$ in S0, in S1 or in other parameterisations, so the parameterisation must be checked before results are compared.

A second trap concerns the scale. The stable law with $\alpha = 2$, $S(2, 0, \gamma, \delta)$, is $N(\delta, 2\gamma^2)$, not $N(\delta, \gamma^2)$. For the Normal distribution, therefore, $\gamma = \sigma/\sqrt{2}$, and a stable $\hat\gamma$ is not directly comparable with a sample standard deviation.

In this course the estimates are reported in S0, and $\delta_1$ is reported separately, so that results can be compared with programs that use S1.

## Three Laws with a Closed-Form Density {h=42509b3f}

Three stable laws have a density in closed form. The Normal distribution is the case $\alpha = 2$, in which $\beta$ plays no role: $\varphi(t) = \exp(-\gamma^2 t^2 + i\delta t)$, which is $N(\delta, 2\gamma^2)$. The Cauchy distribution is the case $\alpha = 1$, $\beta = 0$, with density $f(x) = \gamma/(\pi(\gamma^2 + (x - \delta)^2))$ and $\varphi(t) = \exp(-\gamma|t| + i\delta t)$.

The Lévy distribution is the case $\alpha = 1/2$, $\beta = 1$ (in S1), totally skewed to the right. Its density is

$$ f(x) = \sqrt{\frac{\gamma}{2\pi}}\,(x - \delta)^{-3/2} \exp\Big(-\frac{\gamma}{2(x - \delta)}\Big), \qquad x > \delta. $$

It has a probabilistic interpretation: the first time a Brownian motion hits a given level has a Lévy law.

All other stable densities are computed numerically from $\varphi$, either by direct integration [@Nolan1997] or with the fast Fourier transform [@MittnikDoganogluChenyao1999].

## Normal, Cauchy and Lévy {h=4ff2b6e9}

The chart compares the three closed-form laws: the Normal distribution with variance 1 ($\alpha = 2$, $\gamma = 1/\sqrt{2}$), the Cauchy distribution with $\gamma = 1$ and the Lévy distribution with $\gamma = 1$ in S1. The Cauchy density has a lower peak than the Normal density and much heavier tails: beyond $|x| \approx 2$ its curve stays clearly above the Normal one.

The Lévy density takes only positive values. It rises steeply just above zero and then decays slowly, with a long right tail. Because its index is $\alpha = 1/2$, below 1, its mean is infinite, as the [tails section](#sec-5) explains.

## Worked Example: S1 vs S0 at t = 1 {h=f37cbd4f}

The worked example compares S1 and S0 at a single point, $t = 1$, for $X$ with $\alpha = 1.5$, $\beta = 0.3$, $\gamma = 1$ and $\delta = 0$. A useful value is $\tan\frac{3\pi}{4} = -1$. In S1, @eq:cf-s1 gives $\ln\varphi(1) = -1\cdot[1 - i \cdot 0.3 \cdot (-1)] = -1 - 0.3i$, so $\varphi(1) = e^{-1}(\cos 0.3 - i \sin 0.3) = 0.351 - 0.109i$, with modulus $|\varphi(1)| = e^{-1} = 0.368$.

In S0 the factor $|t|^{1-\alpha} - 1$ is zero at $t = 1$, so the imaginary part vanishes and $\varphi(1) = e^{-1} = 0.368$, a real number. The two characteristic functions have the same modulus. This is what a pure shift does: by $\varphi_{X+b}(t) = e^{ibt}\varphi_X(t)$, a shift multiplies $\varphi$ by a complex number of modulus 1 and leaves the modulus unchanged.

The size of the shift follows from @eq:s0s1: $\beta\gamma\tan\frac{\pi\alpha}{2} = 0.3 \cdot 1 \cdot (-1) = -0.3$. The S1 law with $\delta_1 = 0$ is therefore the S0 law with $\delta_0 = -0.3$.

## Recap: Parameters and the Characteristic Function {h=82e3b3aa}

Stable laws are defined by their characteristic function @eq:cf-s1, and their densities are computed numerically. The four parameters have separate roles: $\alpha$ the tails, $\beta$ the skewness, $\gamma$ the scale and $\delta$ the location.

The S0 and S1 parameterisations differ only in $\delta$, through @eq:s0s1, and scipy uses S1 by default. The closed forms are the Normal distribution ($\alpha = 2$, variance $2\gamma^2$), the Cauchy distribution and the Lévy distribution.

## @end

The characteristic function turns the stability property into a statement about one parameter, $\gamma^\alpha$, which scales linearly with the number of terms. The four parameters separate tails, skewness, scale and location, and the choice between S0 and S1 affects only the location, but in a way that becomes unstable near $\alpha = 1$ in S1. The [next section](#sec-5) studies what $\alpha$ implies for the tails and for the existence of moments.

# Tails and Moments

The index $\alpha$ fixes how fast the tails of a stable law decay and, through the tails, which moments exist. This section derives the power-law tails, the moment condition $p < \alpha$ and their consequences for the measurement of risk.

## Power-Law Tails {h=3b167faa}

For $\alpha < 2$, the tails of a stable law decay like a power of $x$ [@Nolan2020]. As $x \to \infty$,

$$ P(X > x) \sim \gamma^\alpha c_\alpha (1 + \beta)\, x^{-\alpha}, \qquad P(X < -x) \sim \gamma^\alpha c_\alpha (1 - \beta)\, x^{-\alpha}, $$ {#eq:tail}

with $c_\alpha = \Gamma(\alpha)\sin(\pi\alpha/2)/\pi$, where $\Gamma$ is the gamma function and $f \sim g$ means $f/g \to 1$. The skewness $\beta$ enters only through the constants: it changes how much mass each tail carries, not how fast it decays.

On log-log axes, a power-law tail is a straight line with slope $-\alpha$. Doubling the threshold divides the tail probability by $2^\alpha$: for $\alpha = 1.7$ the probability is multiplied by 0.31. The Normal tail behaves differently: it falls like $e^{-x^2/(4\gamma^2)}$, and doubling the threshold makes it practically zero.

When $\beta = \pm 1$ and $\alpha < 1$, one of the tails is missing entirely: the law lives on a half-line. The Lévy distribution of the [previous section](#sec-4) is an example.

## Tails on Log-Log Axes {h=225c97c4}

The chart plots the tail probability $P(X > x)$ of symmetric laws $S(\alpha, 0, 1, 0)$ on log-log axes, for $\alpha = 2$, 1.9, 1.7 and 1.5. The solid lines are the exact tail probabilities and the dotted lines the power laws $c_\alpha x^{-\alpha}$ of @eq:tail. For $\alpha = 1.7$ and 1.5 the solid lines join their dotted asymptotes already near $x = 5$; for $\alpha = 1.9$ they join only further out. The Normal curve bends down and leaves the chart before $x = 10$.

The numbers show the size of the difference. The probability of exceeding 10 is 0.279% for $\alpha = 1.7$, 0.667% for $\alpha = 1.5$, and about $8\times 10^{-11}\%$ for the Normal distribution with the same $\gamma$. A move of ten scale units is a rare but regular event under a stable law with $\alpha = 1.7$ and practically impossible under the Normal distribution.

## Existence of Moments {h=32bb486a}

For $0 < \alpha < 2$, the absolute moment of order $p > 0$, $E|X|^p$, is finite if and only if

$$ p < \alpha. $$ {#eq:moments}

The case $p = 1$ corresponds to the mean and $p = 2$ to the variance. The reason is the tail: the density of a stable law decays like $|x|^{-\alpha-1}$, and $|x|^p$ times such a density is integrable at infinity only if $p < \alpha$.

The consequences follow by comparing $\alpha$ with 1 and 2. For $\alpha < 2$ the variance is infinite, and skewness and kurtosis are not defined. For $1 < \alpha < 2$ the mean exists, and in S1 it equals $\delta_1$. For $\alpha \le 1$ not even the mean exists; the Cauchy and Lévy laws are examples. For $\alpha = 2$, the Normal case, all moments exist.

What do you think? Mandelbrot found $\alpha \approx 1.7$ for cotton. Does the mean daily change exist? Yes, because $1 < 1.7$. The variance, however, does not, because $2 > 1.7$.

## Instability of the Sample Variance {h=3fbde582}

The chart shows what an infinite variance looks like in a sample. It plots the sample variance against the sample size, both on log scales, for a Normal sample (blue), the law $S(2, 0, 1, 0)$ with true variance 2, and for three independent samples from a stable law with $\alpha = 1.7$.

The Normal sample variance converges to its true value 2 and then stays there. The stable paths behave differently: each large draw makes the variance jump, after which it decays slowly until the next large draw. After 20,000 draws the three stable paths end at 10.4, 6.6 and 20.2, with no sign of convergence; a longer sample would only add further jumps.

This is the phenomenon Mandelbrot observed in cotton prices. The sample variance of a stable sample is a number that can always be computed, but it estimates no population quantity.

## Tail Probabilities: Stable vs Normal {h=4be2f1ba}

The table compares the probabilities $P(|X| > k)$ under a stable law with $\alpha = 1.7$ and under the Normal distribution, both with $\beta = 0$, $\gamma = 1$ and $\delta = 0$; the Normal law is therefore $N(0, 2)$. For $k = 2$ the probabilities are close: 15.73% for the Normal and 18.59% for the stable law, a ratio of 1.2. For $k = 3$ they are 3.39% and 7.25%, a ratio of 2.1.

Further out the gap opens. For $k = 5$ the Normal probability is 0.041% and the stable probability 2.13%, a ratio of 52.4. For $k = 10$ the Normal probability is $1.5\times 10^{-10}\%$ and the stable probability 0.56%, a ratio of $3.6\times 10^{9}$.

Near the centre the two laws are almost interchangeable; far in the tails they differ by orders of magnitude. A risk model is judged in the tails, where the choice of law matters most, and a fit that looks good in the centre says little about it.

## Infinite Variance and Risk Measures {h=578986a0}

Several tools of the earlier chapters need a finite variance: the volatility, the Sharpe ratio and its standard error (Chapter 1), mean–variance portfolios, and confidence intervals based on the CLT. With $\alpha < 2$ these numbers can be computed on any sample, but they estimate nothing, as the [previous chart](#s34) showed for the variance.

Other tools survive. Quantiles always exist, so the value at risk is well defined: $\mathrm{VaR}_{1\%} = -q_{1\%}$, the negative of the 1% quantile of returns, is the loss exceeded with probability 1%. The expected shortfall at 2.5%, $\mathrm{ES}_{2.5\%}$, the mean loss beyond the VaR, needs a finite mean and therefore exists only if $\alpha > 1$. The scale $\gamma$ and the interquartile range replace the standard deviation as measures of dispersion.

VaR and ES are developed in [Chapter 10](ch10.html).

## Recap: Tails and Moments {h=95b7b74b}

Stable tails decay as $P(|X| > x) \propto x^{-\alpha}$, a straight line on log-log axes. The absolute moment of order $p$ is finite if and only if $p < \alpha$, so $\alpha < 2$ means an infinite variance.

The sample variance of a stable sample jumps at every large draw and does not settle. Quantiles, the VaR and the scale $\gamma$ keep their meaning.

## @end

The index $\alpha$ is at once the slope of the tails on log-log axes and the threshold for the existence of moments. Under a stable law with $\alpha = 1.7$, an absolute move beyond ten scale units has probability 0.56%, against $1.5\times 10^{-10}\%$ under the Normal distribution, and the sample variance never settles. Measures based on quantiles, such as the VaR, remain valid. These properties also provide the tests of the [critique section](#sec-9): a stable model predicts both a sample variance that does not settle and a specific number of extreme losses. Before fitting the model, the [next section](#sec-6) shows how to simulate it.

# Simulation

Stable laws have no closed-form density, but they are easy to simulate. This section presents the Chambers–Mallows–Stuck (CMS) method, which turns one uniform and one exponential draw into a stable draw, first in the symmetric case and then in the general one.

## The Chambers–Mallows–Stuck Method (Symmetric Case) {h=3805ea7a}

The CMS method [@ChambersMallowsStuck1976] produces a draw of $S(\alpha, 0, 1, 0)$ from two independent variables: a uniform angle $V \sim U(-\pi/2, \pi/2)$ and an exponential variable $W \sim \mathrm{Exp}(1)$ with mean 1. The stable draw is

$$ X = \frac{\sin(\alpha V)}{(\cos V)^{1/\alpha}} \left(\frac{\cos((1 - \alpha)V)}{W}\right)^{(1 - \alpha)/\alpha}, $$ {#eq:cms}

and $\gamma X + \delta \sim S(\alpha, 0, \gamma, \delta)$ gives any scale and location.

The formula can be checked on the two special cases with known laws. For $\alpha = 1$ the exponent $(1 - \alpha)/\alpha$ is zero and $X = \sin V/\cos V = \tan V$; the tangent of a uniform angle is a standard Cauchy variable. For $\alpha = 2$, $\cos((1-\alpha)V) = \cos V$ and $X = \sin(2V)\,(\cos V)^{-1/2}(\cos V/W)^{-1/2} = 2\sin V\sqrt{W}$, which is a $N(0, 2)$ variable, in agreement with the scale convention $S(2, 0, 1, 0) = N(0, 2)$.

The method needs no density, no inversion of a distribution function and no rejection step: each pair $(V, W)$ gives exactly one draw.

## The General Case (β 0) {h=b47bda2a}

For $\beta \ne 0$ and $\alpha \ne 1$, the general formula in the S1 parameterisation [@Weron1996] uses two auxiliary constants that depend only on $\alpha$ and $\beta$:

$$ B = \frac{\arctan(\beta\tan\frac{\pi\alpha}{2})}{\alpha}, \qquad S = \Big(1 + \beta^2\tan^2\tfrac{\pi\alpha}{2}\Big)^{1/(2\alpha)}. $$

The draw is then $X = S\,\sin(\alpha(V + B))/(\cos V)^{1/\alpha} \cdot (\cos(V - \alpha(V + B))/W)^{(1 - \alpha)/\alpha}$, which follows $S(\alpha, \beta, 1, 0; 1)$. For $\beta = 0$, $B = 0$ and $S = 1$, and the formula reduces to @eq:cms.

From the standard draw, any parameters are obtained by scaling and shifting: $\gamma X + \delta_1$ in S1, and $\gamma X + \delta_0 - \beta\gamma\tan\frac{\pi\alpha}{2}$ in S0, using @eq:s0s1. @Weron1996 corrected a formula of the original paper for $\beta \ne 0$; scipy.stats.levy_stable.rvs uses the same method.

## Worked Example: One Draw Step by Step {h=ead98e0e}

The worked example produces one draw with $\alpha = 1.5$ and $\beta = 0$, from a uniform draw $U = 0.8$ and an exponential draw $W = 1.2$. The first step turns $U$, uniform on $(0, 1)$, into a uniform angle on $(-\pi/2, \pi/2)$: $V = \pi(U - 1/2) = 0.942$.

The second step evaluates the factors of @eq:cms: $\sin(\alpha V) = 0.988$, $(\cos V)^{1/\alpha} = 0.702$, and $\big(\cos((1 - \alpha)V)/W\big)^{(1 - \alpha)/\alpha} = 1.104$. The draw is $X = 0.988/0.702 \times 1.104 = 1.554$.

A sample of any size is obtained by repeating the computation with new pairs $(U, W)$. Because each draw is a closed-form function of two standard variables, simulating millions of stable draws takes no longer than simulating Normal ones.

## CMS Draws against the Exact Density {h=8e6672e3}

The chart compares a histogram of 100,000 CMS draws from $S(1.5, 0.5, 1, 0; 1)$ with the exact density, computed numerically from $\varphi$. The histogram follows the density closely, including its asymmetry: with $\beta = 0.5$, the right tail is heavier, and in S1 the mode lies below zero.

The quantiles confirm the agreement in the tails. The sample 1% and 99% quantiles are −5.47 and 9.52; the exact ones, computed by numerical integration with scipy.stats.levy_stable, are −5.39 and 9.79 (the FFT grid of the chart gives −5.40 and 9.76). A two-sample Kolmogorov–Smirnov (KS) test of the CMS draws against draws from levy_stable.rvs gives a statistic of 0.0044 and a p-value of 0.30, so the test finds no difference between the two samples.

## Recap: Simulation {h=6b3e6a74}

The CMS method turns a uniform angle $V$ and an exponential $W$ into one stable draw. The special cases check the formula: $\tan V$ gives the Cauchy distribution and $2\sin V\sqrt{W}$ the Normal distribution $N(0, 2)$.

The skewed case uses the formula of @Weron1996, and a draw in S0 is obtained by shifting with $\beta\gamma\tan\frac{\pi\alpha}{2}$.

## @end

Simulation is the easy part of working with stable laws: each draw is an explicit function of two standard variables, and 100,000 draws reproduce the exact density, including its quantiles. Simulation also supports the rest of the chapter: the Monte Carlo study of the estimators in the [next section](#sec-7) and the simulated variance paths of the [critique section](#sec-9) both rely on it.

# Estimation

Estimating a stable law is harder than simulating one, because the likelihood requires the density, which has no closed form. This section presents the three families of estimators and compares the two most used, McCulloch's quantile method and maximum likelihood.

## Three Families of Estimators {h=fef487d2}

Quantile methods match sample quantiles to those of the stable law. @FamaRoll1971 developed them for symmetric laws, and @McCulloch1986 extended them to all four parameters. They need only sorted data and published tables.

Characteristic-function methods use the empirical characteristic function $\hat\varphi(t) = \frac1n\sum_{j=1}^n e^{itr_j}$ of the returns $r_1, \dots, r_n$. For a stable law $|\varphi(t)|^2 = e^{-2\gamma^\alpha|t|^\alpha}$, so $\ln(-\ln|\varphi(t)|^2) = \ln(2\gamma^\alpha) + \alpha\ln|t|$ is linear in $\ln|t|$. Regressing the empirical version on $\ln|t|$ gives a slope that estimates $\alpha$ [@Koutrouvelis1980].

Maximum likelihood (ML) is the most precise method once the density can be computed [@Nolan2001; @MittnikEtAl1999]. A SAS (Statistical Analysis System) implementation of these estimators is described in @Pele2014.

## McCulloch's Quantile Method {h=41b3b9d7}

McCulloch's method [@McCulloch1986] is built on two ratios of sample quantiles that do not depend on location or scale:

$$ \nu_\alpha = \frac{\hat q_{0.95} - \hat q_{0.05}}{\hat q_{0.75} - \hat q_{0.25}}, \qquad \nu_\beta = \frac{\hat q_{0.95} + \hat q_{0.05} - 2\hat q_{0.5}}{\hat q_{0.95} - \hat q_{0.05}}, $$ {#eq:nu}

where $\hat q_p$ is the sample quantile of order $p$, the value below which a share $p$ of the returns lies. The ratio $\nu_\alpha$ measures the width of the tails relative to the body of the distribution; $\nu_\beta$ measures the asymmetry of the tails around the median.

Tables III and IV of the paper map the pair $(\nu_\alpha, \nu_\beta)$ to estimates $(\hat\alpha, \hat\beta)$. For the Normal distribution $\nu_\alpha = 2.439$, and a larger $\nu_\alpha$ means a smaller $\alpha$. The scale then follows from the interquartile range, $\hat\gamma = (\hat q_{0.75} - \hat q_{0.25})/\phi_3(\hat\alpha, \hat\beta)$, where $\phi_3$ is the interquartile range of $S(\alpha, \beta, 1, 0)$ (Table V), and $\hat\delta_0$ comes from Table VII.

The method is fast and consistent and is valid for $0.6 \le \alpha \le 2$. It is less precise than ML, because it uses only five quantiles, and it is a good starting point for ML.

## McCulloch's Map from ν_α to α {h=f0456455}

The chart draws Table III of @McCulloch1986 for a symmetric sample ($\nu_\beta = 0$): the horizontal axis is $\nu_\alpha$ and the vertical axis the corresponding $\alpha$. The curve starts at $\nu_\alpha = 2.439$ for $\alpha = 2$ and falls as $\nu_\alpha$ grows; heavier tails stretch the 5%–95% range relative to the interquartile range.

The stars place the daily log returns of the four series of the chapter, the indices since 2000 and Bitcoin since 2014. Their values of $\nu_\alpha$ are 3.425 for the BET, 3.338 for the S&P 500, 3.285 for the DAX and 3.803 for Bitcoin. All are well above 2.439, so the tails of all four series are much wider, relative to their bodies, than the Normal distribution allows. Bitcoin lies furthest to the right, with the lowest implied $\alpha$.

## Worked Example: McCulloch on the BET {h=ab51450a}

The worked example applies the method to BET daily log returns in %, from 2000 to 2026, $n = 6{,}670$ observations. The first step computes the five sample quantiles: $\hat q_{0.05} = -1.9345$, $\hat q_{0.25} = -0.5025$, $\hat q_{0.5} = 0.0771$, $\hat q_{0.75} = 0.6649$ and $\hat q_{0.95} = 2.0641$.

The second step forms the ratios of @eq:nu: $\nu_\alpha = 3.9986/1.1674 = 3.425$, and $\nu_\beta = -0.006$, so the sample is practically symmetric. The third step reads Table III in the column $\nu_\beta = 0$, where $\nu_\alpha = 3.2$ corresponds to $\alpha = 1.484$ and $\nu_\alpha = 3.5$ to $\alpha = 1.391$. Linear interpolation between the two rows gives $\hat\alpha \approx 1.414$.

The fourth step uses the remaining tables: the scale is $\hat\gamma = 0.60$ (% a day), the skewness $\hat\beta = -0.011$ and the location $\hat\delta_0 = 0.079$. The estimate $\hat\alpha \approx 1.414$ is far below the Normal value 2.

## Maximum Likelihood {h=d50b37a4}

For a sample $r_1, \dots, r_n$, the log-likelihood is $\ell(\theta) = \sum_{t=1}^n \ln f(r_t; \theta)$, where $\theta = (\alpha, \beta, \gamma, \delta_0)$ is the vector of the four parameters and $f(r_t; \theta)$ the stable density at $r_t$. The ML estimate $\hat\theta$ maximises $\ell$ numerically, starting from the McCulloch estimate. Standard errors come from the curvature of $\ell$ at the maximum: the inverse of the Hessian, the matrix of second derivatives of $\ell$.

The difficulty is the density itself, which must be computed from $\varphi$ by the inversion formula $f(x) = \frac{1}{2\pi}\int e^{-itx}\varphi(t)\,dt$. It can be evaluated on a grid with the fast Fourier transform (FFT) [@MittnikDoganogluChenyao1999] or by direct numerical integration [@Nolan1997]. The function levy_stable.fit performs this computation and needs minutes for a few thousand returns.

In the S0 parameterisation, ML estimation is regular for $\alpha \in (0, 2)$: the estimator is asymptotically Normal and its standard errors shrink like $1/\sqrt{n}$ [@Nolan2001].

## McCulloch vs ML in a Monte Carlo Study {h=9da34224}

The Monte Carlo study draws 200 samples from $S(1.7, 0, 1, 0)$ for each sample size, $n = 500$ and $n = 2500$, and estimates $\alpha$ by both methods. The box plots show the distribution of $\hat\alpha$ across the 200 samples; the dashed line marks the true value 1.7. Both estimators are centred on 1.7, so neither is visibly biased.

The precision differs. The standard deviation of $\hat\alpha$ is 0.1065 for McCulloch and 0.0611 for ML at $n = 500$, and 0.0462 against 0.0298 at $n = 2500$. ML is therefore more precise on the same data: McCulloch's standard deviation is 1.74 times that of ML at $n = 500$ and 1.55 times at $n = 2500$. With $n = 500$, McCulloch's method occasionally reaches the bound $\alpha = 2$, visible as outliers at the top of the first box.

## Recap: Estimation {h=efe44d30}

McCulloch's method needs five quantiles, two ratios and four tables; it is fast and robust. ML computes the density numerically from $\varphi$ and maximises the likelihood numerically; it is the most precise.

Every estimate must state its parameterisation, S0 or S1, and the units of $\gamma$ and $\delta$, since both change when returns are expressed in % instead of decimals.

## @end

The two estimators complement each other: McCulloch's quantile method gives a quick and robust estimate from five quantiles, and ML, started from it, gains a factor of about 1.7 in precision on samples from $S(1.7, 0, 1, 0)$. On BET daily returns the quantile method already gives $\hat\alpha \approx 1.414$, far from the Normal value 2. The [next section](#sec-8) fits the stable law by ML to all four series and confronts it with the data and with two alternative laws.

# Fitting Real Returns

This section fits the stable law by maximum likelihood to daily log returns of four series: the BET, the S&P 500 and the DAX from 2000 to 2026, and Bitcoin from 2014 to 2026. The stable fit is compared with two alternatives: the Normal distribution and the Student-t distribution. The comparison uses densities, QQ plots, tail probabilities, counts of large losses, the VaR and an information criterion.

## Stable Fits of Daily Log Returns {h=b9e22a7f}

The table reports the estimates for the four series. Returns are daily log returns in %; the indices cover 2000–2026 and Bitcoin 2014–2026, with seven trading days a week. The column "McC." gives McCulloch's $\hat\alpha$; the ML estimates are in S0, with standard errors (SE) in parentheses; $\hat\gamma$, $\hat\delta_0$ and $\hat\delta_1$ are in % a day.

The ML estimates of $\alpha$ are 1.486 (SE 0.020) for the BET, with $n = 6{,}670$ days; 1.540 (SE 0.021) for the S&P 500, with $n = 6{,}714$; 1.606 (SE 0.021) for the DAX, with $n = 6{,}786$; and 1.410 (SE 0.026) for Bitcoin, with $n = 4{,}384$. The ML estimates of $\beta$ are −0.016 (SE 0.034) for the BET, −0.181 (SE 0.036) for the S&P 500, −0.179 (SE 0.040) for the DAX and 0.019 (SE 0.036) for Bitcoin.

The scale $\hat\gamma$ is 0.62% a day for the BET, 0.58% for the S&P 500, 0.74% for the DAX and 1.54% for Bitcoin. The locations differ between the two parameterisations exactly as @eq:s0s1 predicts: for the S&P 500, $\hat\delta_0 = 0.093$ and $\hat\delta_1 = 0.000$, the difference being $\hat\beta\hat\gamma\tan\frac{\pi\hat\alpha}{2}$.

## Reading the Estimates {h=c1c93cac}

All four estimates of $\alpha$ are far below 2. For the S&P 500 the distance from the Normal case is $(2 - 1.540)/0.021 \approx 22$ standard errors, so the Normal value is rejected by any conventional test. The heaviest tails are those of Bitcoin, with $\hat\alpha = 1.410$.

The skewness is negative for the S&P 500 and the DAX, about −0.18 for both: the left tail is heavier, which reflects crashes. For the BET and Bitcoin, $\hat\beta$ lies within two standard errors of 0, so these series show no significant asymmetry.

The scale shows how different the series are in size: $\hat\gamma$ is 1.54% a day for Bitcoin and 0.58% for the S&P 500. The sample standard deviations, 3.49% and 1.21%, are larger than $\sqrt{2}\hat\gamma$, the standard deviation of a Normal law with the same $\gamma$, because they are inflated by the tails. Finally, McCulloch's method gives a lower $\hat\alpha$ than ML for every series: the two methods weigh the tails differently, the former through five quantiles, the latter through every observation.

## Fitted Densities on a Log Scale (1/2) {h=83bdc95c}

The chart plots the empirical density of daily returns (dots, from a histogram) for the BET and the S&P 500 on a log scale, together with three fitted laws: Normal (red), Student-t (green) and stable, fitted by ML (blue). The log scale makes the tails visible; on a linear scale they would be indistinguishable from zero.

The Normal density is a parabola on this scale and fails on both counts: it is too low at the centre, and it falls below the data beyond about 4% in either direction, so it predicts far too few large returns. The Student-t and stable densities both follow the body of the distribution closely, from the peak down to densities of about $10^{-2}$.

## Fitted Densities on a Log Scale (2/2) {h=e8ac5fc6}

The same comparison for the DAX and Bitcoin shows the far tails more clearly. Beyond the range where both heavy-tailed laws fit, the stable density stays above the data points, while the Student-t density is closer to them. For Bitcoin, whose returns reach about ±20% on the chart, the gap is visible in both tails.

This is the first sign of the pattern found throughout the section: the stable law describes the centre and the moderate tails well, but its power-law tails with $\alpha$ near 1.5 are too heavy for the most extreme returns.

## QQ Plots against the Three Models (1/2) {h=f7648b02}

A QQ (quantile–quantile) plot puts the empirical quantiles of the returns on the vertical axis against the quantiles of a fitted model on the horizontal axis. If the model is correct, the points lie on the 45-degree line. Deviations at the ends show which tail is mis-specified and in which direction.

For the BET and the S&P 500, the three models agree in the centre, where all points lie on the line. They separate at the ends. The Normal points (red) turn steeply away from the line: the empirical extremes are much larger than the Normal quantiles. The stable points (blue) turn the other way and become flat: the stable model places its extreme quantiles far beyond the data, at about −23% for the BET where the empirical quantile is about −10%. The Student-t points (green) stay closest to the line.

## QQ Plots against the Three Models (2/2) {h=5052eed5}

The DAX and Bitcoin show the same three patterns. The Normal distribution has tails that are too short, which appears as steep ends; the stable distribution has tails that are too long, which appears as flat ends, with model quantiles far beyond the data; the Student-t distribution is the closest of the three. For Bitcoin the most extreme stable quantiles reach about 70% in absolute value, against empirical quantiles near 17%.

A QQ plot of this kind identifies the failure of each model without any test: the Normal model underestimates and the stable model overestimates the extremes.

## The Left Tail on Log-Log Axes (1/2) {h=ea660694}

The chart looks at the left tail on log-log axes. The points show, for each loss level $x$ (in %, horizontal axis), the share of days with a loss above $x$; the lines show the same probability under each fitted model. The dotted horizontal line marks the probability of one day in the sample, the smallest non-zero empirical frequency.

For the BET and the S&P 500, all three models agree with the data for losses up to about 2–3%. Beyond that the Normal curve (red) falls vertically. The stable curve (blue) continues as a straight line, as @eq:tail predicts. The data points bend down below it, and the Student-t curve (green) follows them further.

## The Left Tail on Log-Log Axes (2/2) {h=ea5dc63e}

For the DAX and Bitcoin the picture is the same. The empirical tail bends down faster than the stable line: on log-log axes, its slope becomes steeper than $-\hat\alpha$. In terms of the tail index, the far tail of the data is lighter than a stable tail with index $\hat\alpha$, so its tail index is larger than $\hat\alpha$.

The ML estimate of $\alpha$ is driven mostly by the many observations in the body and the moderate tails. It cannot adjust to the few extreme observations, because a stable law ties the slope of the far tail to the same $\alpha$ that shapes the centre.

## Counting Large Losses {h=63c4c0f3}

The table turns the tail probabilities into numbers of days. For each model, the expected number of days with a loss beyond $x$ is $n \times P(r < -x)$ under the fitted law. For the S&P 500, 23 days with a loss above 5% were observed; the Normal model expects 0.1 days, the Student-t 27.0 and the stable law 57.3. For losses above 10%, one day was observed, against practically zero, 4.5 and 19.1 days. For losses above 15% none was observed, against practically zero, 1.5 and 10.2.

The other series confirm the pattern. For the BET, 6 days with losses above 10% were observed, against practically zero under the Normal model, 8.2 under the Student-t and 22.8 under the stable law. For Bitcoin, 56 days with losses above 10% were observed, against 8.1, 54.2 and 73.9; and 4 days with losses above 20%, against practically zero, 12.3 and 26.9.

The Normal model predicts almost no large losses. The stable law predicts too many, and the gap grows with the threshold. The Student-t is closest to the observed counts. For the S&P 500, the stable fit expects 19.1 days below −10% in 6,714 days, where one was observed.

## VaR 1% and VaR 0.1% under the Three Models {h=9110e0b9}

The value at risk at level $p$ is $\mathrm{VaR}_p = -q_p$, the daily loss exceeded with probability $p$ (Chapter 10). The level is written $p$ here, because $\alpha$ already denotes the index of stability. The table compares the empirical VaR, from the sample quantile, with the VaR of the three fitted models, at the levels 1% and 0.1%.

At the 1% level, the Normal model gives values that are too low and the stable model values that are too high. For the S&P 500, the empirical VaR 1% is 3.5%; the Normal model gives 2.8%, the Student-t 3.5% and the stable law 4.5%. For Bitcoin the empirical value is 10.5%, against 8.0%, 11.1% and 14.3%.

At the 0.1% level the stable model overshoots much more. Its VaR 0.1% is roughly two to four times the empirical quantile: 23.1% against 9.5% for the BET, 19.7% against 7.2% for the S&P 500, 20.0% against 7.1% for the DAX, and 74.2% against 18.1% for Bitcoin. The Normal model, at 3.7% for the S&P 500, is about half the empirical value.

## Comparison of the Three Models {h=e15ee800}

The Akaike information criterion (AIC) is $2k - 2\ell(\hat\theta)$, where $k$ is the number of parameters and $\ell(\hat\theta)$ the maximised log-likelihood; a lower value is better. It penalises extra parameters, so a model with more parameters must improve the likelihood enough to be preferred.

Both heavy-tailed models beat the Normal distribution on every series: the Student-t by 1,306 (DAX) to 2,411 (BET) AIC points, the stable law by 1,204 to 2,319; for the S&P 500, the AIC is 21,650 for the Normal model, 19,702 for the Student-t and 19,794 for the stable law. The Student-t is preferred for every series, even though it has fewer parameters than the stable law. For the S&P 500 its advantage is 92 points.

The estimated degrees of freedom $\hat\nu$ of the Student-t are 2.43 for the BET, 2.67 for the S&P 500, 3.09 for the DAX and 2.23 for Bitcoin, all between 2 and 3.1. A Student-t law has finite moments only of order below $\nu$: these estimates imply a finite variance, but almost no finite moments of higher order.

## Recap: Fitting Real Returns {h=b9025c31}

On all four series, the ML estimates of $\alpha$ lie between 1.410 and 1.606, far from 2. The stable law describes the body of the distribution much better than the Normal distribution.

It overstates the far tails: it predicts too many extreme losses and gives a VaR 0.1% that is too high. By the AIC, the Student-t distribution fits better on every series.

## @end

Fitted to real returns, the stable law is a large improvement on the Normal distribution and a worse description than the Student-t. Its $\hat\alpha$ between 1.410 and 1.606 captures the heavy tails near the centre, but the same $\alpha$ forces power-law tails that are too heavy far out: 19.1 expected days with losses beyond 10% for the S&P 500 against one observed, and a VaR 0.1% two to four times the empirical one. The Student-t estimates $\hat\nu$ between 2.23 and 3.09 point to a finite variance. The [next section](#sec-9) tests the infinite variance directly.

# Critique of the Stable Model: Finite or Infinite Variance

The fits of the previous section suggest that the far tails of returns are lighter than any stable tail. This section tests the infinite-variance hypothesis directly, with three tests drawn from the theory of the chapter, and then explains why daily returns nevertheless look stable.

## Test 1: Stability of α under Aggregation {h=2d3fd6ce}

The first test uses the stability property itself. If daily returns were i.i.d. stable, weekly and monthly sums would be stable with the same $\alpha$. If instead daily returns have a finite variance, the GCLT predicts that the estimated $\alpha$ rises towards 2 as returns are summed over longer horizons.

The chart reports ML estimates of $\alpha$ with 95% confidence intervals for daily, weekly and monthly returns of the four series; the dashed line marks $\alpha = 2$. For the S&P 500 the estimates rise from 1.54 (daily) to 1.64 (weekly) and 1.82 (monthly). For Bitcoin they rise from 1.41 to 1.67 and then to 2.00, at the bound 2: for monthly Bitcoin returns the stable fit reaches the Normal case.

## Reading the Aggregation Test {h=f98819e1}

The estimated $\alpha$ rises with the horizon for the S&P 500, the DAX and Bitcoin. For the DAX the sequence is $1.61 \to 1.72 \to 1.69$, so the rise occurs between daily and weekly returns. For the S&P 500 the 95% intervals of the daily and the monthly estimates do not overlap, so the rise is statistically clear.

The BET is the exception: $1.49 \to 1.56 \to 1.50$, without a clear rise. Its monthly sample is short, 321 months, and the interval of the monthly estimate is wide, so the test has little power for this series.

The same pattern appears in the classic studies of @Officer1972 and @AkgirayBooth1988. A rising $\alpha$ is exactly what the GCLT predicts for sums of returns with a finite variance that converge slowly to the Normal distribution: in that world the stable fit to daily data is an approximation that the aggregation gradually removes.

## Test 2: Convergence of the Sample Variance {h=d76cacb4}

The second test repeats Mandelbrot's observation on modern data. The chart plots the running sample variance of S&P 500 daily log returns (in %, thick blue line) from 2000 to 2026, on a log scale. The variance jumps in 2008 and 2020, during the two largest crises of the sample, and then settles near 1.47.

The thin lines are paths of the same length simulated from the fitted stable law. They behave as in the [simulation of the tails section](#s34): repeated jumps, slow decay and no convergence. Their final variances lie between 5.4 and 15.6, far above the value of the data. The observed variance is stable in a way that a stable law with $\hat\alpha = 1.540$ does not allow.

## Test 3: Thickness of the Far Tail {h=6c065ac1}

The stable $\alpha$ is determined by the whole distribution, mostly by its body, where most observations lie. The tail index of the far tail alone is estimated separately, with the Hill estimator [@Hill1975], which uses only the largest observations; Chapter 5 develops it.

For stock indices the far tail decays roughly like $x^{-3}$, the "inverse cubic law" [@GopikrishnanEtAl1999]. A tail index near 3 means a finite variance, and no stable law has it, since every non-Normal stable law has a tail index below 2.

The counts of large losses in the [previous section](#s58) agree: the fitted stable laws expect many more losses beyond 10% than were observed. @LauLauWingender1990 add further evidence against the stable model, from the behaviour of sample moments.

## Origin of the Apparent Stability of Daily Returns {h=e4edb5aa}

Two explanations account for the stable appearance of daily returns. The first is volatility clustering (Chapters 2 and 9): calm and turbulent periods alternate. A mixture of Normal laws with changing variance has heavy tails and a finite variance [@BlattbergGonedes1974]. If a single stable law is fitted to data generated by such a mixture, the estimate is $\hat\alpha < 2$, even though no observation comes from an infinite-variance law.

The second explanation is offered by truncated and tempered stable laws. They behave like stable laws at moderate scales but have lighter tails in the extreme region, so all their moments are finite. @MantegnaStanley1995 fitted a truncated Lévy flight to the S&P 500. @GrabchakSamorodnitsky2010 show that such laws explain why returns look stable at short horizons and Normal at long ones, which is the pattern of the aggregation test.

## Conclusion of the Critique {h=8e6206f4}

Mandelbrot was right about the facts. Daily returns have heavy tails, and the Normal distribution describes the data very poorly; every fit in this chapter confirms it.

The infinite variance, however, is not supported. The estimated $\alpha$ rises with aggregation, the sample variance of real returns settles, and the far tail is lighter than any stable tail. All three tests point the same way.

Stable laws remain useful in several roles: as a benchmark for heavy tails, as a tool for simulation, through the idea of scaling across horizons, and for data whose variance is truly infinite, such as some insurance losses and some crypto tokens. The GCLT remains a reminder that rules built on $\sqrt{n}$ require a finite variance. [Chapter 5](ch5.html) estimates the tail index directly, with extreme value theory.

## Recap: The Critique {h=93b2d9dc}

Stability predicts the same $\alpha$ at every horizon, but the data show an $\alpha$ that rises with the horizon. The sample variance of real returns settles, while that of stable samples does not.

Volatility clustering and tempered tails explain the apparent $\alpha < 2$ of daily returns. The verdict of the chapter is: heavy tails, yes; infinite variance, no.

## @end

The three tests reject the infinite variance without rejecting heavy tails. Aggregation raises $\hat\alpha$ towards 2, as the GCLT predicts for finite-variance returns; the running variance of the S&P 500 settles near 1.47, while stable paths end between 5.4 and 15.6; and the far tail decays roughly like $x^{-3}$, a tail index outside the stable range. Volatility clustering and tempered tails reconcile these results with the stable appearance of daily data.

# AI for Scientific Discovery

The tools of the chapter can be applied to questions that are still open. This section formulates one such question, about the tails of Bitcoin, describes how AI (artificial intelligence) tools could support a study of it, lists the checks that the study needs, and gives a project seed.

## An Open Research Question {h=09fc87b0}

The open question is whether the tails of Bitcoin returns have become lighter as the market matured. The market gained more participants, futures contracts from 2017 and spot exchange-traded funds (ETFs) from 2024. If the tails have become lighter, risk models fitted on old data overstate today's risk.

The question remains open for two reasons. The estimated $\alpha$ moves with volatility regimes, so a change in $\hat\alpha$ may reflect a calmer period rather than a lighter tail. In addition, yearly samples are short, and the estimates have wide intervals.

On the full sample, Bitcoin has $\hat\alpha = 1.410$ (SE 0.026), the lowest of the four series of the chapter. AI tools can speed up a study of this kind; they do not replace checking it [@WangEtAl2023].

## Possible Uses of AI Tools {h=24c9f878}

AI tools can contribute at four stages. For the literature, they can list papers on tail estimates for crypto assets and summarise their methods. For the code, they can draft a rolling-window McCulloch estimator with bootstrap intervals. For robustness, they can propose other windows, ML instead of quantiles, or the Student-t $\nu$ instead of $\alpha$. For the explanation, they can write a first draft of the interpretation of a table of yearly estimates.

The example prompt on the slide asks for exactly the second contribution: a Python function that estimates the stable $\alpha$ of daily log returns on rolling 365-day windows with @McCulloch1986 and adds 95% bootstrap intervals. Each of these outputs is a draft, and the next frame lists what must be checked before it is used.

## What to Check {h=8c95cad5}

The parameterisation comes first: the code must state whether it uses S0 or S1; scipy uses S1 by default, and the location differs by $\beta\gamma\tan\frac{\pi\alpha}{2}$. The units come second: with returns in % or in decimals, $\gamma$ and $\delta$ change, while $\alpha$ and $\beta$ do not.

Overlapping windows are not independent, so rolling estimates must not be read as separate tests; a trend in a rolling $\hat\alpha$ can arise from a few extreme days that stay in many windows. Numerical results must be recomputed for at least one window with a second method, ML or the quantile tables. Finally, every cited paper must exist, and its DOI must be checked.

## Project Idea {h=a8a8cd50}

The project asks whether the tail of Bitcoin returns is lighter now than in 2014–2018. The data are the daily closing prices of Bitcoin from the course data, 2014–2026.

The steps follow the chapter. Estimate $\alpha$ by McCulloch's method and by ML for each calendar year, with bootstrap intervals. Repeat the estimation on returns divided by a rolling volatility, to see whether the trend survives once volatility clustering is removed. Compare the results with the Student-t $\hat\nu$ of each year.

The deliverable is one table, one chart and a paragraph on what the data can and cannot show. Any use of AI must be declared, together with a list of the AI errors that were corrected.

## @end

The question about Bitcoin's tails uses every tool of the chapter: both estimators, the S0/S1 distinction, the link between volatility clustering and apparent heavy tails, and the Student-t comparison. AI tools can draft the code and the literature review, but the checks on parameterisation, units, overlapping windows and references remain the researcher's task.

# Summary

The summary collects the formulas of the chapter and three questions for self-assessment.

## Key Formulas {h=3ef8e6ce}

The table gathers the formulas of the chapter. Stability is the property @eq:stable with the scale rule $c^\alpha = a^\alpha + b^\alpha$ @eq:scale, and the sum of $n$ copies has the scale factor $n^{1/\alpha}$. The characteristic function in S1 is @eq:cf-s1, and the locations in S0 and S1 are linked by @eq:s0s1. The Normal case is $S(2, \beta, \gamma, \delta) = N(\delta, 2\gamma^2)$.

The tails decay as in @eq:tail, $P(X > x) \sim \gamma^\alpha c_\alpha(1 + \beta)x^{-\alpha}$, and for $\alpha < 2$ the absolute moment of order $p$ is finite if and only if $p < \alpha$ @eq:moments. The CMS formula @eq:cms simulates the symmetric case, and McCulloch's ratios @eq:nu are the starting point of estimation.

## Self-Assessment {h=60f48d85}

The first question: daily returns are i.i.d. $S(1.5, 0, \gamma, 0)$; by how much does the scale grow from 1 day to 20 days? By the factor $20^{1/1.5} = 7.37$, which is 1.65 times the factor $\sqrt{20}$ of the square-root-of-time rule.

The second question: a fitted law has $\alpha = 1.2$; do the mean and the variance exist? By @eq:moments, the mean exists because $1 < 1.2$, and the variance does not, because $2 > 1.2$.

The third question: scipy reports $\delta = 0.05$ with $\alpha = 1.7$, $\beta = -0.2$ and $\gamma = 0.6$; what is $\delta_0$? Scipy uses S1, so @eq:s0s1 gives $\delta_0 = 0.05 + (-0.2)(0.6)\tan(0.85\pi) = 0.05 + 0.061 = 0.111$. The next chapter, [Chapter 4](ch4.html), turns to probability for finance.

## @end

Stable laws are the only laws closed under summation and the only limits of normalised sums. They describe the heavy tails of daily returns far better than the Normal distribution, but their infinite variance is contradicted by aggregation, by the behaviour of the sample variance and by the far tails. [Chapter 5](ch5.html) estimates the far tail directly with extreme value theory.
