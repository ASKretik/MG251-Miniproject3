# Mean–Variance Portfolios and Factor Models — report draft

*MG251 mini project. Draft generated from the notebook outputs; the team should edit the wording and check every
number against `report/tables/` (each table there is written by a named cell in `notebooks/main.ipynb`).
Target ≤ 6 pages with the figures listed.*

---

## 1. Data and setup (½ page)

We use 17 US stocks across ten sectors (AAPL, MSFT, INTC, JPM, C, JNJ, PFE, XOM, KO, WMT, HD, F, CAT, UNP, NEE,
NEM, VZ), month-end adjusted closes from Yahoo Finance, January 2006 – December 2025 (240 months). Adjusted
closes include dividends and splits, so monthly `pct_change` is a total simple return. Factors (Mkt-RF, SMB, HML,
RF, Mom, the FF5 factors) and the 25 size × book-to-market and 10 momentum portfolios come directly from Ken
French's data library (percent converted to decimals; only the monthly value-weighted block is parsed).

Train = 2006–2015 (contains the 2008 crisis), test = 2016–2025 (contains COVID). Everything used to choose
weights is estimated on train data only, enforced by an assertion in the code.

Sanity checks: no missing months; SPY's correlation with Mkt-RF + RF is 0.995; Mkt-RF averages 10.2% a year.
Five monthly moves exceed ±50% (C and F in 2008–09); we checked them by hand and they are real.

*Caveat — survivorship:* all 17 firms still exist in 2026. We deliberately included laggards (C, INTC, F, PFE),
but Section 8.6 of the notebook shows how much hindsight can flatter a backtest.

## 2. Week 1: the frontier, out-of-sample, and why the optimiser disappoints (1½ pages)

**Means are the weak link.** Over 120 train months the 95% confidence interval for a stock's annual mean is
roughly ±10 to ±30 percentage points; 10 of the 17 intervals include zero (Fig. 01). The median relative
standard error is **65% for means but only 7% for volatilities**. Pinning a mean down to ±2% with a 20% volatility
would need about **384 years** of data. The optimiser needs 17 means + 153 covariances = 170 inputs.

**In sample the optimiser looks brilliant** (Fig. 02). No random long-only portfolio reaches the frontier. The
shorts-allowed max-Sharpe portfolio promises an annual Sharpe of **1.55**, against SPY's 0.47 over the same
months. Its weights are extreme: 33% KO, 32% HD, 26% UNP, −27% C, −11% INTC, with gross exposure of 2.37. It
correlates only 0.44 with SPY. Under the CAPM it should be the market portfolio; it isn't, because it is
chasing noise in μ.

**Out of sample the promise collapses** (Figs. 03, 04). Train weights are frozen, the portfolio is rebalanced
monthly, and the test is 2016–2025:

| Strategy | Predicted Sharpe | Realised Sharpe | Realised vol | Final $1 |
|---|---|---|---|---|
| GMV (shorts) | 0.94 | 0.92 | 12.7% | 3.67 |
| GMV (long only) | 0.74 | 0.99 | 12.9% | 4.06 |
| Max-Sharpe (shorts) | **1.55** | **0.73** | 16.6% | 3.62 |
| Max-Sharpe (long only) | 1.15 | 0.97 | 14.9% | 4.67 |
| Equal weight 1/N | 0.64 | 0.98 | 14.4% | 4.53 |
| SPY | — | 0.85 | 15.1% | 3.95 |

With the halves swapped (train 2016–25, test 2006–15) the pattern repeats and is starker: max-Sharpe with
shorts promises **2.02 and delivers 0.58**, below 1/N (0.64). GMV with shorts delivers 0.99. Averaged over both
orderings, max-Sharpe (shorts) misses its promise by 1.13 Sharpe points; GMV and 1/N miss by less than 0.1.

**Why.** The optimiser is an *error maximiser*. Raising KO's mean by one standard error (4.9% a year, an
entirely plausible estimation error) moves **110%** of the max-Sharpe portfolio. Raising KO's volatility by one
standard error moves only 17% (Fig. 13). Resampling the train months 200 times makes the max-Sharpe point
wander across the chart, while the GMV point, which uses no means, stays in a tight cloud. Its volatility varies
3.6× less (Fig. 12). So the problem is mainly μ, and GMV holds up because it never uses μ.

## 3. Week 2: beta, the CAPM, and the French portfolios (1½ pages)

**Our 17 stocks.** Time-series betas range from 0.29 (NEM) to 1.90 (C); OLS betas match cov/var exactly. Across
stocks, average excess return is almost flat in beta (Fig. 05): the fitted slope is **2.8% per unit of beta
vs the CAPM's 10.2%**, with R² = 0.05. Using betas from 2006–15 to predict 2016–25 returns gives the same picture
(slope 2.4% vs 13.0%). This is the "too flat" security market line that Black, Jensen and Scholes documented.
Seventeen stocks make a weak test, though: each average return has a standard error of about 5% a year, and
errors in beta bias the slope towards zero. So we move to portfolios.

**French portfolios, July 1963 – Sep 2026 (759 months).**

| Test assets | Model | Mean \|α\| (ann.) | Avg R² | GRS | p |
|---|---|---|---|---|---|
| 25 size × B/M | CAPM | 2.32% | 0.73 | 4.17 | <0.001 |
| | FF3 | 1.04% | 0.91 | 3.63 | <0.001 |
| | FF3 + Mom | 0.97% | 0.91 | 3.20 | <0.001 |
| 10 momentum | CAPM | 2.7% | 0.78 | 5.01 | <0.001 |
| | FF3 | 3.3% | 0.80 | 5.40 | <0.001 |
| | FF3 + Mom | 1.45% | 0.89 | 3.67 | <0.001 |

The CAPM's errors line up with characteristics (Figs. 06, 07): small-value portfolios earn +5.6% a year of
CAPM alpha and small-growth −6.4%. FF3 halves the average error and removes the value pattern, because SMB and
HML are built from the same sorts. FF3 makes the momentum portfolios *worse*, since value and momentum are
negatively correlated (−0.19). Only the momentum factor fixes them: past losers' alpha goes from −12.6% to −3.2%.
Small-growth (S1B1, −5.6% under FF3) stays mispriced under every model. GRS rejects all three models on the long
sample, where T = 759 gives the test the power to detect 1% alphas. On 2006–2025 (T = 240) the statistics are much
smaller (CAPM 1.87, FF3 1.80 on the 25 portfolios) but still reject at the 5% level.

**Risk, mispricing or luck?** SMB raises small-stock R² from 67% to 95%, so it captures a lot of co-movement.
Yet it earned 3.4% a year before Banz (1981) published the size effect (t = 2.2) and only 0.2% since (t = 0.2;
Fig. 17). A premium that disappears after publication looks like luck or a mispricing that was traded away.
A factor can still matter for *risk* without paying a premium.

## 4. Factor covariance in the optimiser (1 page)

We replace the sample Σ with Σ = BΣ_FBᵀ + D, estimated on train data with 1, 3 and 4 factors, and add
Ledoit–Wolf shrinkage as a benchmark. The key assumption is that residuals are uncorrelated across stocks. It
roughly holds: the average residual correlation is **0.002**, against 0.30 for raw returns. It breaks for
specific pairs (JPM/C 0.27, KO/JNJ 0.30).

| Comparison (shorts-allowed GMV unless noted) | Sample Σ | FF3 Σ |
|---|---|---|
| Static GMV realised vol, forward / swapped | 12.7% / 11.6% | 12.7% / 11.2% |
| Static long-only GMV realised vol, forward / swapped | 12.9% / 11.9% | 12.3% / 11.3% |
| Bootstrap 95% range of each GMV weight (average) | 23% | **12%** |
| Rolling 60-month GMV: monthly turnover | 30% | **11%** |
| Rolling 60-month GMV: realised vol, 2011–2025 | 14.7% | **13.0%** |
| Rolling 60-month GMV: realised Sharpe | 0.97 | 0.84 |
| Covariance parameters, N = 17 / 500 | 153 / 125,250 | 74 / 2,006 |

With 120 months for 17 stocks the sample Σ is already decent, so the static gain is small. Where estimation error
bites — short windows, resampling — the factor structure halves the uncertainty in each weight, cuts turnover by
almost two-thirds, and lowers realised risk (Figs. 08, 09, 10). Ledoit–Wolf achieves similar risk reduction.
Honest caveat: the lower-risk rolling factor portfolio also had a lower Sharpe over 2011–2025; GMV minimises
risk, not return.

**μ vs Σ.** In a 2 × 3 grid of max-Sharpe test Sharpe ratios, changing only μ or only Σ to the factor version
barely helps. Changing both raises max-Sharpe from 0.73 to 0.89 (forward) and from 0.58 to 0.77 (swapped). Even
so, GMV with the sample Σ, which uses no μ at all, is as good or better (0.92 / 0.99). Dropping μ beats every
attempt to estimate it.

## 5. What surprised us (½ page)

- How large the gap between promise and reality is: a Sharpe of 2.02 became 0.58.
- That naive 1/N matched or beat every optimised portfolio except GMV.
- That a one-standard-error change in one stock's mean reshuffles more than the whole portfolio.
- That survivorship can matter more than any optimiser: an equal-weight portfolio of today's ten largest US
  companies turns $1 into $168 over 2006–2025. The ten largest *as of January 2006* turn it into $5.6, behind SPY's
  $7.9 (Fig. 16).
- That DFSVX's apparent CAPM underperformance (α = −3.8%) is almost entirely factor exposure: its FF3 α is −0.5%,
  with SMB loading 0.80, HML loading 0.63 and R² of 0.98 (Fig. 14).

## 6. Why do we need factor models? (1 page)

Our numbers make three arguments.

1. **To explain expected returns.** The CAPM leaves systematic, characteristic-linked pricing errors: an average
   |α| of 2.3% a year on the 25 size/value portfolios, and a security market line one-quarter as steep as theory
   predicts. Adding SMB and HML halves the errors; adding momentum fixes the momentum portfolios. Factor models
   are a much better description of *why* average returns differ.
2. **To reduce estimation error.** A full covariance matrix needs N(N+1)/2 numbers: 153 for our stocks and 125,250
   for 500 stocks, which is not even estimable from 20 years of monthly data. A 3-factor model needs 74 and 2,006.
   Imposing structure adds a little bias but removes a lot of variance. In our data it halves the bootstrap
   uncertainty of each GMV weight.
3. **To build stable portfolios that behave as promised.** With factor Σ, rolling GMV turnover falls from 30% to
   11% a month and realised risk falls from 14.7% to 13.0%. Combining factor-implied μ with factor Σ is the only
   version of max-Sharpe that clearly improves out of sample.

The deeper lesson is that expected returns are the hardest thing in finance to estimate. Factor models do not
make μ easy, but they turn 170 free parameters into a few interpretable ones, and that is what makes
mean–variance optimisation usable in practice.

### Figures referenced
01 mean CI · 02 frontier · 03/03b OOS wealth · 04/04b predicted vs realised · 05 beta vs return ·
06/06b actual vs predicted · 07/07b alpha heatmap · 08 weight stability · 09 rolling weights ·
10 GMV vol by estimator · 11 diversification · 12 wobbling frontier · 13 error maximiser · 14 fund alpha ·
15 factor correlations · 16 survivorship · 17 SMB
