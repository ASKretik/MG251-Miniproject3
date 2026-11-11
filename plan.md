# MG251 Mini Project: Mean–Variance Portfolios and Factor Models

## How to use this file

Put this file in the root of an empty project folder, open Claude Code there, and start with:

> Read plan.md. Do Phase 0 only, then stop and summarise what you did.

Then work phase by phase ("Do Phase 1", "Do Phase 2", ...). Each phase has a **Done when** checklist and an **Understand this** list. The brief says every team member must be able to explain every line of code and every number reported, so don't move on until the team can answer the Understand-this questions.

## Rules for Claude (working agreement)

1. **Work one phase at a time.** At the end of a phase, stop, list the files changed and the key numbers produced, and wait.
2. **Readable over clever.** Plain numpy/pandas with short comments explaining *why*, not *what*. When a library does the maths (statsmodels OLS, cvxpy), also compute the key quantity once by hand (e.g. beta = cov/var) and assert the two agree.
3. **No look-ahead.** Anything used to choose weights must be estimated on train data only. Add an assertion where train/test meet.
4. **Units.** Store returns as monthly *simple* returns in decimals. French data comes in percent; divide by 100 on load. Annualise only for display: mean × 12, vol × √12, Sharpe × √12.
5. **Cache downloads** in `data/raw/`, but the notebook must still run end to end from an empty `data/` folder.
6. **Don't add dependencies** beyond `requirements.txt` without asking.
7. Every figure is saved to `figures/` as PNG (dpi 200) with a title, labelled axes with units, and a one-line caption printed under it in the notebook.
8. If a data source fails (URL changed, ticker missing), stop and report; don't silently substitute.

## Project structure

```
mg251-miniproject/
├── plan.md
├── README.md               # how to run, team, data sources
├── requirements.txt
├── config.py               # tickers, dates, paths, constants (single source of truth)
├── src/
│   ├── data.py             # yfinance + French library download, parsing, alignment
│   ├── estimation.py       # means, covariances, standard errors, bootstrap
│   ├── optimise.py         # GMV, max-Sharpe, frontier (with/without shorts)
│   ├── backtest.py         # out-of-sample evaluation, metrics
│   ├── factors.py          # time-series regressions, GRS test, factor covariance
│   └── plots.py            # all plotting functions
├── notebooks/
│   └── main.ipynb          # runs download → final chart, calls src/
├── data/{raw,processed}/
├── figures/
├── tests/                  # small sanity tests (pytest)
└── report/                 # report draft + slides
```

`requirements.txt`: numpy, pandas, scipy, matplotlib, statsmodels, scikit-learn, cvxpy, yfinance, requests, jupyter, nbconvert, pytest.

(Do not use `pandas_datareader`; it is unreliable on recent Python/pandas. Download the French CSV zips directly.)

## Default choices (change in `config.py` if the team decides otherwise)

| Item | Default | Why |
|---|---|---|
| Market | US | French factors are US; IIMA library for India is a robustness extension |
| Frequency | Monthly | Matches French data; less noise than daily |
| Period | Jan 2006 – Dec 2025 (240 months) | Check the French library's latest month and trim if needed |
| Train / test | 2006–2015 / 2016–2025 | First half / second half, as in the brief. Train contains 2008; test contains 2020 |
| Stocks (17) | AAPL, MSFT, INTC (tech); JPM, C (financials); JNJ, PFE (health); XOM (energy); KO, WMT (staples); HD, F (discretionary); CAT, UNP (industrials); NEE (utilities); NEM (materials); VZ (telecom) | Different sectors, full history since 2006, deliberately includes some poor performers (C, INTC, F, PFE) to soften "picked today's winners" bias |
| Risk-free | `RF` column from French factors file (1-month T-bill) | Already monthly, already aligned |
| Market for regressions | French `Mkt-RF` | Total-return, value-weighted, all US stocks |
| Market index for comparisons | SPY (adjusted close) via yfinance | Investable index, includes dividends |
| Prices | Adjusted close (`auto_adjust=True`) | Includes splits and dividends |

## Phase 0: Setup

**Tasks**
- Create the folder structure, `requirements.txt`, `config.py`, `README.md`, empty `src/` modules with docstrings.
- `config.py` holds: `TICKERS`, `SECTORS` (dict ticker → sector), `START`, `END`, `TRAIN_END`, `MARKET_ETF = "SPY"`, paths, `SEED = 251`, `N_BOOT = 500`.
- Create `notebooks/main.ipynb` with section headers matching the phases below.

**Done when**
- `pip install -r requirements.txt` works and `python -c "import config"` runs.

## Phase 1: Data (Week 1)

**Tasks**
1. `src/data.py::load_prices(tickers, start, end)`: download daily adjusted closes with yfinance, take the last price of each month (`resample("ME").last()`), compute `pct_change()`. Index as `pd.Period` monthly (`to_period("M")`) so it joins cleanly with French data.
2. `load_french(name)`: download `https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip`, unzip in memory, parse **only the first (monthly, value-weighted) block**. The files have text headers and then an "Annual" section further down; stop at the first blank line after the monthly table. Convert `YYYYMM` to Period, replace `-99.99`/`-999` with NaN, divide by 100. Files needed:
   - `F-F_Research_Data_Factors` (Mkt-RF, SMB, HML, RF)
   - `F-F_Momentum_Factor` (Mom)
   - `F-F_Research_Data_5_Factors_2x3` (for the factor-correlation visual)
   - `25_Portfolios_5x5` (size × book-to-market)
   - `10_Portfolios_Prior_12_2` (momentum deciles)
   If a filename 404s, check the library page for the current name and report.
3. Build one tidy dataset: stock returns, SPY return, French factors, aligned on common months. Save to `data/processed/monthly.parquet` (or CSV).
4. Sanity checks (print, and add `tests/test_data.py`):
   - No missing months for any ticker in the sample window.
   - Flag any monthly return with |r| > 50% and print it for manual inspection.
   - Correlation of SPY return with `Mkt-RF + RF` should be > 0.98.
   - Annualised mean of `Mkt-RF` over the sample should look plausible (roughly 5–12%).

**Outputs**: `data/processed/monthly.*`, a table of per-stock annualised mean, vol, min, max.

**Done when**: checks pass and the summary table is in the notebook.

**Understand this**
- Why adjusted close and not close? What would a dividend-paying stock look like without adjustment?
- Why monthly simple returns rather than log returns for portfolio maths?
- What is survivorship bias, and how does choosing these 17 tickers in 2026 affect the results?

## Phase 2: Estimation and confidence

**Tasks**
1. On **train data**: sample mean vector μ and sample covariance Σ (monthly).
2. Confidence in each mean: standard error = σ/√T; annualised 95% CI. Plot a bar chart of annualised means with error bars (`figures/01_mean_ci.png`).
3. Confidence in covariances: bootstrap months (B = 500, seed from config) and report the 95% interval for each volatility and for a few correlations. Show that relative uncertainty in vols/correlations is far smaller than in means.
4. Print the number of parameters estimated: N means + N(N+1)/2 covariance terms.

**Done when**: the CI chart exists and there is a short table comparing "relative SE of mean" vs "relative SE of vol" per stock.

**Understand this**
- Why is the mean so poorly estimated even with 120 months? (SE of the mean shrinks with √T, not with sampling frequency.)
- Roughly how many years of data would you need to pin a 10% annual mean down to ±2%?

## Phase 3: Efficient frontier

**Tasks** (all on train data, `src/optimise.py`)
1. **Random portfolios**: 20,000 long-only weight vectors from a Dirichlet(1,...,1); plot vol vs mean, coloured by Sharpe.
2. **Proper optimiser** (cvxpy):
   - Global minimum variance (GMV): min wᵀΣw s.t. Σw = 1, with and without w ≥ 0.
   - Frontier: minimise variance for a grid of target returns, with and without w ≥ 0.
   - Max-Sharpe, **short selling allowed**: closed form w ∝ Σ⁻¹(μ − r_f), normalised to sum to 1.
   - Max-Sharpe, **long only**: solve min yᵀΣy s.t. (μ − r_f)ᵀy = 1, y ≥ 0, then w = y / Σy. (This turns the non-convex Sharpe ratio into a convex problem.)
   - Verify the closed-form GMV w = Σ⁻¹1 / 1ᵀΣ⁻¹1 matches cvxpy's unconstrained result.
3. Plot: random cloud, both frontiers, individual stocks, GMV and max-Sharpe points, the capital allocation line from r_f through the tangency portfolio (`figures/02_frontier.png`).
4. Table of weights for the four portfolios (GMV / max-Sharpe × shorts / no shorts).

**Done when**: frontier plot and weights table exist; closed-form vs solver check passes.

**Understand this**
- Why does the random cloud never reach the frontier?
- Why are the short-allowed max-Sharpe weights often extreme (large longs and shorts)? Is that a bug or a finding?
- Connect to the CAPM: under its assumptions, the max-Sharpe portfolio should be the market. Is yours?

## Phase 4: Out-of-sample test vs equal weights

**Tasks** (`src/backtest.py`)
1. Freeze weights estimated on train data. Apply them to test months, rebalancing monthly back to the fixed weights.
2. Strategies: GMV (shorts / no shorts), max-Sharpe (shorts / no shorts), equal weight 1/N, SPY.
3. Metrics on test data, annualised: mean, vol, Sharpe, max drawdown, final wealth of $1.
4. **Predicted vs realised**: next to each strategy's realised test Sharpe and vol, show what the train data predicted. This gap is the main result of Week 1.
5. Plots: cumulative wealth on test (`figures/03_oos_wealth.png`); predicted vs realised Sharpe bar chart (`figures/04_pred_vs_real.png`).
6. Robustness: swap the halves (train on 2016–2025, test on 2006–2015) and report the same table.

**Done when**: the results table exists for both orderings.

**Understand this**
- Why does the "optimal" portfolio usually lose to 1/N? (Estimation error in μ; the optimiser is an "error maximiser" that overweights stocks whose means were overestimated by luck.)
- Why does GMV usually hold up much better than max-Sharpe out of sample? (It doesn't use μ at all.)

## Phase 5: Beta and the CAPM on your stocks (Week 2)

**Tasks** (`src/factors.py`)
1. For each stock, full sample: regress excess return on `Mkt-RF` (statsmodels OLS, Newey–West SEs with 6 lags). Record α, β, t-stats, R². Check β by hand as cov(rᵢ, m)/var(m).
2. Cross-section: scatter of β (x) vs average excess return (y) for the 17 stocks.
   - Draw the **CAPM line**: intercept 0, slope = average `Mkt-RF`.
   - Draw the **fitted line** from regressing average excess return on β.
   (`figures/05_beta_vs_return.png`)
3. Report R² of the cross-sectional fit and the slope vs the CAPM slope.
4. Repeat with betas estimated on train and average returns on test (betas estimated *before* the returns they're supposed to predict).

**Done when**: plot and table exist.

**Understand this**
- What does it mean that the fitted line is flatter than the CAPM line? (Same pattern as Black's figures in the textbook.)
- Why is a cross-section of 17 stocks a weak test? Why does finance test on portfolios instead? (Noise in individual stock means, errors in betas.)
- Expected vs actual returns: why can't we conclude the CAPM is wrong from one sample?

## Phase 6: Is beta the whole story? French portfolios

**Tasks**
1. Test assets: the 25 size × book-to-market portfolios and the 10 momentum portfolios. Use **July 1963 – latest** (the standard sample), and also report the project window 2006–2025.
2. For each portfolio, time-series regressions of excess returns on:
   - CAPM: `Mkt-RF`
   - FF3: `Mkt-RF`, `SMB`, `HML`
   - FF3 + momentum (Carhart): add `Mom`
3. For each model, **predicted** average excess return = loadings × average factor returns; **actual** = average excess return. Pricing error = α.
4. Plots: actual vs predicted, one panel per model, with a 45° line, points labelled (e.g. "S1B5" = smallest, highest B/M) (`figures/06_actual_vs_predicted.png`). A 5×5 heatmap of α under CAPM and FF3 (`figures/07_alpha_heatmap.png`).
5. Summary table per model: mean |α|, root-mean-square α, average R², and the **GRS test**:
   GRS = ((T − N − L)/N) · α̂ᵀ Σ̂⁻¹ α̂ / (1 + f̄ᵀ Ω̂⁻¹ f̄) ~ F(N, T − N − L),
   where N = number of portfolios, L = number of factors, Σ̂ = residual covariance (εᵀε/T), Ω̂ = factor covariance (divisor T), f̄ = mean factor returns. Implement it in `factors.py` with a docstring explaining each term.
6. Identify the portfolios CAPM gets most wrong (expected: small-value high, small-growth low, momentum winners/losers) and whether FF3 / Carhart fix them.

**Done when**: three-model comparison table and both figures exist for both samples.

**Understand this**
- What does the GRS test's null hypothesis say in plain words? (All α = 0 jointly.)
- Why does FF3 fix size/value but not momentum?
- Is a factor premium a reward for risk, a mispricing, or luck? What evidence could separate them? (Out-of-sample/post-publication decay; behaviour in recessions; data snooping.)

## Phase 7: Factor-model covariance back in the optimiser

**Tasks**
1. On **train data only**, regress each stock's excess returns on the factors to get loadings B (N × K) and residual variances (diagonal D). Factor covariance Σ_F = sample covariance of factors.
   Σ_factor = B Σ_F Bᵀ + D.
   Build three versions: 1-factor (market only), FF3, FF3 + Mom.
2. Extra benchmark: Ledoit–Wolf shrinkage (`sklearn.covariance.LedoitWolf`).
3. Rerun Phase 4 with each covariance estimator. The cleanest comparison is **GMV realised test volatility**, because GMV doesn't depend on μ.
4. **Separating μ from Σ**: a 2 × 3 grid of max-Sharpe out-of-sample Sharpe ratios:
   - Rows: sample Σ, FF3 Σ
   - Columns: sample μ, factor-implied μ (B × mean factor returns), no μ (GMV)
   This answers "is the problem the expected returns or the covariances?"
5. **Weight stability**:
   - Bootstrap the train months (B = 500), recompute GMV weights under sample Σ vs factor Σ, plot the distribution of each stock's weight (box plot, `figures/08_weight_stability.png`).
   - Rolling 60-month windows across the full period: average monthly turnover Σ|wₜ − wₜ₋₁| under each estimator.
6. Parameter count table: sample cov N(N+1)/2 vs factor model NK + K(K+1)/2 + N, for N = 17, 20, 500 and K = 1, 3, 4. (For N = 20: 210 vs 86 with K = 3. For N = 500: 125,250 vs 2,006.)

**Done when**: OOS table across estimators, μ/Σ grid, stability figure, turnover numbers, parameter table.

**Understand this**
- What assumption makes Σ = BΣ_FBᵀ + D valid? (Residuals uncorrelated across stocks.) Check it: show the average residual correlation in your data. Same-sector pairs (JPM/C, XOM?) are where it breaks.
- Why does imposing structure (bias) reduce estimation error (variance) and help out of sample?

## Phase 8: Original visualisations / experiments (bonus marks)

The brief requires at least two of the team's own design. Build at least three and pick the best for the presentation. Each goes in `src/plots.py` + its own notebook section with a 2–3 sentence "what this shows" note.

Candidates (the first three in the brief's spirit, the rest less obvious):
1. **Diversification curve**: portfolio vol vs number of stocks (random equal-weight subsets of your 17, averaged over many draws), overlaid with theory σ√(ρ + (1 − ρ)/n) for ρ = 0, 0.3, 0.6. The floor at σ√ρ is the undiversifiable (market) risk.
2. **Wobbling frontier**: 50 bootstrap frontiers overlaid, plus clouds of GMV and max-Sharpe points. The GMV cloud is tight; the max-Sharpe cloud is everywhere.
3. **Error-maximiser experiment**: nudge one stock's mean by +1 standard error and show how much the max-Sharpe weights move; then nudge its volatility by +1 SE and compare. Shows μ errors dominate.
4. **Fund alpha decomposition**: regress a real fund (e.g. a small-cap value fund such as DFSVX, or BRK-B) on CAPM, FF3, FF3 + Mom; show α shrinking as factors are added (waterfall chart). Links to the "CAPM α 4%, FF3 α 0%" question.
5. **Factor correlation heatmap** (FF5 + Mom), full sample and rolling 10-year. HML vs CMA correlation shows near-redundant factors. Explain why factors should ideally be close to uncorrelated.
6. **Survivorship experiment**: compare a portfolio of the 10 largest US companies *as of Jan 2006* (e.g. XOM, GE, MSFT, C, BAC, WMT, PG, JNJ, AIG, PFE; verify the list) with the 10 largest *today*, both held over 2006–2025. Shows what picking today's winners does to backtests.
7. **Factor with movement but no premium**: show SMB explains a lot of small-stock return variation (R²) while its average return since ~1980 is near zero. Or use a French industry portfolio.

**Done when**: at least three finished figures with notes; team has chosen one for the presentation.

## Phase 9: Questions worth chasing (stretch)

Answer these in the report where the work above gives evidence. Most are already set up by earlier phases:
- Why does the optimal portfolio lose to 1/N, and is it μ or Σ? → Phases 4 and 7 (grid).
- How many numbers for 20 vs 500 stocks? → Phase 7 parameter table.
- Should max-Sharpe look like the index? → Phase 3 weights vs SPY / market-cap weights.
- Risk, mispricing, or luck? → Phase 6 plus pre/post-publication split (e.g. size effect before vs after 1981).
- Factor that moves returns without a premium → Phase 8 #7.
- Sensitivity to period/stocks → Phase 4 swapped halves, Phase 8 #6. Optional: rerun Phases 3–4 on 15 Indian stocks (NSE tickers ending `.NS`) with IIMA factors.

## Phase 10: Assemble deliverables

**Notebook**
- `main.ipynb` runs top to bottom from an empty `data/` folder: `jupyter nbconvert --to notebook --execute notebooks/main.ipynb`. Fix anything that fails.
- Every section starts with a markdown cell: what we do, and why.
- Run `pytest` and make sure it passes.

**Report (≤ 6 pages)** — written by the team; Claude can draft from the notebook outputs but the team edits.
1. Data and setup (½ page)
2. Week 1: frontier, out-of-sample, why the optimiser disappoints (1½ pages)
3. Week 2: beta, CAPM vs FF3 on French portfolios (1½ pages)
4. Factor covariance in the optimiser (1 page)
5. What surprised us (½ page)
6. **"Why do we need factor models?"** (1 page) — argue from your own numbers: CAPM pricing errors (Phase 6), parameter count and estimation error (Phase 7), stability and OOS risk (Phase 7).

**Presentation (5 minutes, recorded)**
- Built around the single best visualisation. Suggested arc: the puzzle (optimiser loses to 1/N) → why (estimation error, picture) → the fix (factor structure) → what it bought us (one number).

## Suggested team split (4 people)

| Person | Owns | Also reviews |
|---|---|---|
| A | Phases 1–2 (data, estimation) | Phase 5 |
| B | Phases 3–4 (optimiser, backtest) | Phase 7 |
| C | Phases 5–6 (CAPM, French portfolios, GRS) | Phase 2 |
| D | Phase 7 + presentation | Phase 4 |
| All | Phase 8 (one visual each), report | — |

Everyone must be able to explain every phase, so each person presents their phase's code to the others before Phase 10.

## Common pitfalls checklist

- [ ] French data in percent → divided by 100?
- [ ] Monthly vs annual blocks in French CSVs → only the monthly block parsed?
- [ ] yfinance dates vs French `YYYYMM` → both converted to monthly Periods before joining?
- [ ] Any train-period estimate leaking into test-period weights?
- [ ] Excess returns (minus RF) used in regressions and Sharpe ratios?
- [ ] Annualisation: mean × 12, vol × √12, never vol × 12?
- [ ] Short-allowed weights reported honestly even if extreme?
- [ ] Random seed fixed for bootstraps and random portfolios?
- [ ] Every number in the report traceable to a notebook cell?
