# MG251 Mini Project: Mean–Variance Portfolios and Factor Models

We build mean–variance portfolios from 17 US stocks (2006–2025), test them out of sample
against 1/N, then use the CAPM and Fama–French factor models to explain why the optimiser
disappoints and whether a factor-model covariance fixes it. `plan.md` has the full phase-by-phase plan.

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

pytest                                    # sanity tests
jupyter nbconvert --to notebook --execute notebooks/main.ipynb   # full pipeline
```

The notebook runs end to end from an empty `data/` folder (about one minute). Downloads are cached in `data/raw/`.
Delete those files to force a fresh download. If you move the project folder, recreate `.venv` (venvs store absolute
paths): `python3 -m venv --clear .venv && .venv/bin/pip install -r requirements.txt`.

## Headline results

| Question | Answer from our data |
|---|---|
| Does the optimiser win out of sample? | No. Max-Sharpe (shorts) promised Sharpe 1.55 and delivered 0.73; 1/N delivered 0.98. Swapped halves: 2.02 → 0.58 |
| Why? | Means are noise: median relative SE 65% for means vs 7% for vols. A 1-SE nudge to one mean moves 110% of the portfolio |
| Does beta explain returns? | Barely: cross-sectional slope 2.8% vs CAPM 10.2%, R² 0.05 |
| CAPM vs FF3 on 25 size/value portfolios (1963–) | Mean \|α\| 2.32% → 1.04%; momentum needs Carhart (3.30% → 1.45%) |
| What does factor Σ buy? | Bootstrap weight range 23% → 12%; rolling GMV turnover 30% → 11%/month, realised vol 14.7% → 13.0% |

Full numbers: `report/tables/` (written by the notebook). Report draft: `report/report_draft.md`.
Video script: `report/video_script.md`.

## Layout

| Path | Contents |
|---|---|
| `config.py` | Tickers, dates, paths and constants (single source of truth) |
| `src/data.py` | yfinance and French library download, parsing, alignment |
| `src/estimation.py` | Means, covariances, standard errors, bootstrap |
| `src/optimise.py` | GMV, max-Sharpe, efficient frontier (with/without shorts) |
| `src/backtest.py` | Out-of-sample evaluation and metrics |
| `src/factors.py` | Time-series regressions, GRS test, factor covariance |
| `src/plots.py` | All plotting functions |
| `notebooks/main.ipynb` | Runs download → final chart, calling `src/` |
| `figures/` | Saved figures (PNG, 200 dpi) |
| `tests/` | pytest sanity checks |
| `report/` | Report draft, video script, and `tables/` with every number quoted |

## Data sources

- **Stock and ETF prices**: Yahoo Finance via `yfinance`, adjusted close (`auto_adjust=True`), month-end.
- **Factors and test portfolios**: Kenneth French Data Library,
  <https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html>
  (`F-F_Research_Data_Factors`, `F-F_Momentum_Factor`, `F-F_Research_Data_5_Factors_2x3`,
  `25_Portfolios_5x5`, `10_Portfolios_Prior_12_2`).

## Team

| Person | Owns |
|---|---|
| _name_ | Phases 1–2 (data, estimation) |
| _name_ | Phases 3–4 (optimiser, backtest) |
| _name_ | Phases 5–6 (CAPM, French portfolios, GRS) |
| _name_ | Phase 7 + presentation |
