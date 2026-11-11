"""Out-of-sample evaluation: apply frozen train weights to test months and compute metrics.

Weights are fixed and the portfolio is rebalanced back to them every month, so the
portfolio return in month t is simply R_t @ w.
"""
import numpy as np
import pandas as pd

from src import optimise


def portfolio_returns(weights, returns):
    """Monthly returns of fixed-weight portfolios. `weights` is a DataFrame (stocks x strategies)."""
    return returns[weights.index] @ weights


def max_drawdown(r):
    """Largest peak-to-trough fall in the wealth path of returns `r` (a negative number)."""
    wealth = (1 + r).cumprod()
    return (wealth / wealth.cummax() - 1).min()


def metrics(port_rets, rf):
    """Annualised mean, vol, Sharpe (on excess returns), max drawdown and final wealth of $1."""
    rows = {}
    for name, r in port_rets.items():
        ex = r - rf
        rows[name] = {
            "ann_mean": r.mean() * 12,
            "ann_vol": r.std() * np.sqrt(12),
            "sharpe": ex.mean() / ex.std() * np.sqrt(12),
            "max_drawdown": max_drawdown(r),
            "final_wealth": (1 + r).prod(),
        }
    return pd.DataFrame(rows).T


def predicted_metrics(weights, mu, sigma, rf):
    """What the train estimates said each strategy would deliver (annualised)."""
    rows = {}
    for name, w in weights.items():
        m, v, s = optimise.portfolio_stats(w.values, mu, sigma, rf)
        rows[name] = {"pred_mean": m * 12, "pred_vol": v * np.sqrt(12), "pred_sharpe": s * np.sqrt(12)}
    return pd.DataFrame(rows).T


def strategy_weights(mu, sigma, rf):
    """The four optimised portfolios plus 1/N, all from train inputs."""
    w = optimise.four_portfolios(mu, sigma, rf)
    w["Equal weight (1/N)"] = 1 / len(w)
    return w


def run(train, test, tickers, market, sigma=None, mu=None):
    """Full Phase 4 evaluation for one train/test split.

    `sigma` and `mu` default to the train sample estimates; Phase 7 passes other estimators.
    Returns (weights, realised test returns, results table with predicted vs realised).
    """
    # No look-ahead: nothing used to pick weights may come from the evaluation months.
    # (The swapped-halves robustness check trains on the later half, so test for overlap, not order.)
    assert train.index.intersection(test.index).empty, "train and test months overlap"
    rf_train = train["RF"].mean()
    mu = train[tickers].mean() if mu is None else mu
    sigma = train[tickers].cov() if sigma is None else sigma

    weights = strategy_weights(mu, sigma, rf_train)
    rets = portfolio_returns(weights, test[tickers])
    rets[market] = test[market]

    realised = metrics(rets, test["RF"])
    predicted = predicted_metrics(weights, mu, sigma, rf_train)
    table = realised.join(predicted)
    return weights, rets, table


def rolling_weights(df, tickers, estimator, window=60, long_only=False):
    """GMV weights re-estimated each month from the previous `window` months.

    `estimator(train_slice) -> Sigma`. Weights for month t use months t-window..t-1 only.
    Returns (weights DataFrame indexed by the month they are held, realised GMV returns).
    """
    months = df.index
    rows, rets = {}, {}
    for i in range(window, len(months)):
        past = df.iloc[i - window:i]
        w = optimise.gmv(estimator(past), long_only=long_only)
        rows[months[i]] = w
        rets[months[i]] = df[tickers].iloc[i].to_numpy() @ w
    W = pd.DataFrame(rows, index=tickers).T
    return W, pd.Series(rets)


def turnover(W):
    """Average monthly turnover: mean over t of sum_i |w_it - w_i,t-1|."""
    return W.diff().abs().sum(axis=1).iloc[1:].mean()


def mu_sigma_grid(train, test, tickers, factor_cols, long_only=False):
    """Max-Sharpe out-of-sample Sharpe ratios for every (Sigma, mu) combination.

    Rows: sample Sigma, factor Sigma. Columns: sample mu, factor-implied mu (B @ mean factors),
    no mu (GMV). Answers: is the problem the expected returns or the covariances?
    """
    from src import estimation as est
    from src import factors as fm

    assert train.index.intersection(test.index).empty
    ex = est.excess_returns(train, tickers)
    rf = train["RF"].mean()
    sigmas = {"Sample Σ": train[tickers].cov(),
              "Factor Σ": fm.factor_covariance(ex, train[factor_cols])[0]}
    # Both mu versions are total (not excess) returns, so add back the average T-bill rate.
    mus = {"Sample μ": train[tickers].mean(),
           "Factor μ": fm.factor_implied_mu(ex, train[factor_cols]) + rf}
    out = pd.DataFrame(index=list(sigmas), columns=list(mus) + ["No μ (GMV)"], dtype=float)
    for s_name, S in sigmas.items():
        for m_name, m in mus.items():
            w = pd.DataFrame({"w": optimise.max_sharpe(m, S, rf, long_only=long_only)}, index=tickers)
            out.loc[s_name, m_name] = metrics(portfolio_returns(w, test[tickers]), test["RF"]).loc["w", "sharpe"]
        w = pd.DataFrame({"w": optimise.gmv(S, long_only=long_only)}, index=tickers)
        out.loc[s_name, "No μ (GMV)"] = metrics(portfolio_returns(w, test[tickers]), test["RF"]).loc["w", "sharpe"]
    return out
