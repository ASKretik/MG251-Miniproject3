# 5-minute video script (4 speakers × ~75 seconds)

**Built around one visualisation: the wobbling frontier (Fig 12).** It opens the video, carries the explanation,
and comes back at the end when we show the fix (Fig 08 is the same resampling experiment applied to the factor model).

`[Show: …]` lines are screen cues, not spoken. Replace [A]–[D] with names. Each part is ~180–190 words,
about 75 seconds at a natural pace.

---

## Speaker A — The hook and the data (0:00 – 1:15)
*Owns Phases 1–2*

[Show: Fig 12, the wobbling frontier, full screen]

Hi, we're [team name]. This picture is our whole project in one image. Every faint blue curve is an "efficient
frontier": the best portfolios you can build from the same seventeen stocks. The only thing that changes from curve
to curve is a random reshuffle of the same ten years of history. If optimal portfolios were reliable, these curves
would sit on top of each other. They don't, and the next five minutes explain why.

[Show: list of the 17 stocks]

Our data is twenty years of monthly prices, 2006 to 2025, for seventeen US stocks across ten sectors, from Yahoo
Finance, plus factor data from Kenneth French's library. We estimate on the first ten years and test on the second ten.

[Show: Fig 01, mean returns with confidence intervals]

The root problem is here. Each bar is a stock's average return with a 95% confidence interval. Ten of seventeen
intervals include zero. The typical error is 65% of the mean, but only 7% of the volatility. Pinning a mean down to
plus or minus two percent would take almost four hundred years of data. Over to [B].

---

## Speaker B — What the optimiser does with noise (1:15 – 2:30)
*Owns Phases 3–4*

[Show: Fig 12 again, pointing at the orange and green dots]

Back to our picture. Each orange dot is the maximum-Sharpe portfolio from one reshuffle of history, and they're
scattered everywhere. Each green dot is the minimum-variance portfolio, which ignores expected returns entirely, and
those stay in a tight cluster. Same stocks, same data, but one portfolio is stable and the other isn't. The difference
is the means.

[Show: Fig 13, error maximiser]

To prove it, we nudged Coca-Cola's expected return up by one standard error, well within the noise. 110% of the
portfolio moved. The same nudge to its volatility moved 17%. The optimiser is an error maximiser: it bets hardest on
whichever means were overestimated by luck.

[Show: Fig 04, predicted vs realised Sharpe]

So what happens out of sample? Max-Sharpe with short selling promised a Sharpe ratio of 1.55 and delivered 0.73,
the worst of everything we tried. Simply holding all stocks equally got 0.98. With the halves swapped it promised 2.02
and delivered 0.58. The minimum-variance portfolio, the tight green cluster, kept its promise. [C] will show what factor
models add.

---

## Speaker C — From CAPM to factor models (2:30 – 3:45)
*Owns Phases 5–6*

[Show: Fig 05, beta vs average return]

If we can't trust each stock's mean, maybe theory can tell us what it should be. The CAPM says expected return depends
only on beta. The orange line is the CAPM's prediction for our stocks; the dashed line is what we found. The slope is
2.8% instead of 10.2%, and beta explains only 5% of the differences. Seventeen stocks is a noisy test, though, so we
used Fama and French's 25 size-and-value portfolios and 10 momentum portfolios, back to 1963.

[Show: Fig 07, CAPM vs FF3 alpha heatmap]

This shows each portfolio's CAPM pricing error, and it isn't random. Small value stocks earn 5.6% a year more than
the CAPM predicts; small growth stocks earn 6.4% less. Adding size and value factors, the Fama–French three-factor
model, cuts the average error from 2.3% to about 1%. Momentum portfolios need a fourth factor, which shrinks past
losers' error from minus 12.6% to minus 3.2%.

No model is perfect: the GRS test rejects all three. But a few factors explain far more than beta alone. [D] shows
what that does for our wobbling optimiser.

---

## Speaker D — The fix: factors in the optimiser (3:45 – 5:00)
*Owns Phase 7*

[Show: parameter count, 125,250 vs 2,006]

Factor models also simplify risk. Instead of estimating every covariance separately, we assume stocks move together
only through a few factors. For five hundred stocks, that's about two thousand numbers instead of a hundred and
twenty-five thousand.

[Show: Fig 08, bootstrap weight stability]

Here's the same reshuffling experiment as our opening picture, now looking at portfolio weights. Orange uses the raw
covariance matrix, blue the three-factor model. The blue boxes are half as wide: the average uncertainty in each weight
falls from 23 to 12 percentage points.

[Show: Fig 09, rolling weights]

Re-estimating every month on five-year windows, turnover falls from 30% to 11% a month, and realised risk drops from
14.7% to 13%. To be honest, that safer portfolio also had a slightly lower Sharpe ratio; minimum variance targets risk,
not return.

[Show: Fig 12 one last time]

So, back to the wobbling frontier. Optimisers fail because they trust noisy numbers. Factor models explain returns
the CAPM can't, and they replace thousands of noisy numbers with a few meaningful ones. That's why finance moved to
factor models. Thanks for watching.

---

### Recording tips
- Rehearse with a timer. If a part runs long, trim the figure description rather than the numbers.
- Keep Fig 12 on screen whenever it's mentioned; it is the thread that ties the four parts together.
- Every number is in `report/tables/09_headline_numbers.csv` or the report, so each speaker can explain it if asked.
- The brief requires everyone to understand every number, so each speaker should also rehearse one other part.
