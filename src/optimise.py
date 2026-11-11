"""Portfolio optimisation: random portfolios, GMV, efficient frontier, max-Sharpe.

Inputs are monthly: `mu` (mean simple returns), `sigma` (covariance), `rf` (mean T-bill rate).
Closed-form solutions are checked against cvxpy wherever a closed form exists.
"""
import cvxpy as cp
import numpy as np
import pandas as pd

import config


def _arr(x):
    return np.asarray(x, dtype=float)


def _psd(sigma):
    """Wrap a covariance for cvxpy's quad_form (symmetrised, flagged PSD)."""
    s = _arr(sigma)
    return cp.psd_wrap((s + s.T) / 2)


def portfolio_stats(w, mu, sigma, rf=0.0):
    """Monthly mean, vol and Sharpe ratio of weights `w`."""
    w = _arr(w)
    m = w @ _arr(mu)
    v = np.sqrt(w @ _arr(sigma) @ w)
    return m, v, (m - rf) / v


def random_portfolios(mu, sigma, n=config.N_RANDOM_PORTFOLIOS, seed=config.SEED, rf=0.0):
    """Long-only weights drawn uniformly on the simplex (Dirichlet(1,...,1)) and their stats."""
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(len(_arr(mu))), size=n)
    means = W @ _arr(mu)
    vols = np.sqrt(np.einsum("ij,jk,ik->i", W, _arr(sigma), W))
    return pd.DataFrame({"mean": means, "vol": vols, "sharpe": (means - rf) / vols})


# --- Global minimum variance ---------------------------------------------

def gmv_closed_form(sigma):
    """w = Sigma^-1 1 / (1' Sigma^-1 1): the unconstrained (shorts allowed) GMV."""
    x = np.linalg.solve(_arr(sigma), np.ones(len(sigma)))
    return x / x.sum()


def gmv(sigma, long_only=False):
    """Minimise w' Sigma w subject to sum(w) = 1 (and w >= 0 if long_only)."""
    n = len(sigma)
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1] + ([w >= 0] if long_only else [])
    cp.Problem(cp.Minimize(cp.quad_form(w, _psd(sigma))), cons).solve(solver=cp.CLARABEL)
    out = np.array(w.value).ravel()
    if long_only:
        out = np.clip(out, 0, None)  # strip solver noise like -1e-10
        out /= out.sum()
    return out


# --- Max-Sharpe (tangency) ------------------------------------------------

def max_sharpe_closed_form(mu, sigma, rf):
    """Shorts allowed: w is proportional to Sigma^-1 (mu - rf), rescaled to sum to 1."""
    x = np.linalg.solve(_arr(sigma), _arr(mu) - rf)
    return x / x.sum()


def max_sharpe_long_only(mu, sigma, rf):
    """Long only. Sharpe isn't convex, so solve min y' Sigma y s.t. (mu - rf)'y = 1, y >= 0,
    then w = y / sum(y). Scaling y doesn't change the Sharpe ratio, which is why this works.
    """
    n = len(sigma)
    y = cp.Variable(n)
    cons = [(_arr(mu) - rf) @ y == 1, y >= 0]
    cp.Problem(cp.Minimize(cp.quad_form(y, _psd(sigma))), cons).solve(solver=cp.CLARABEL)
    out = np.clip(np.array(y.value).ravel(), 0, None)
    return out / out.sum()


def max_sharpe(mu, sigma, rf, long_only=False):
    return max_sharpe_long_only(mu, sigma, rf) if long_only else max_sharpe_closed_form(mu, sigma, rf)


# --- Frontier -------------------------------------------------------------

def frontier(mu, sigma, targets, long_only=False):
    """Minimum variance for each target mean. Returns DataFrame of (mean, vol).

    Long-only targets outside [min(mu), max(mu)] are infeasible and skipped.
    """
    mu_a, n = _arr(mu), len(sigma)
    w = cp.Variable(n)
    target = cp.Parameter()
    cons = [cp.sum(w) == 1, mu_a @ w == target] + ([w >= 0] if long_only else [])
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, _psd(sigma))), cons)
    rows = []
    for t in targets:
        if long_only and not (mu_a.min() <= t <= mu_a.max()):
            continue
        target.value = t
        prob.solve(solver=cp.CLARABEL)
        if w.value is not None and prob.status == "optimal":
            rows.append((t, np.sqrt(max(prob.value, 0))))
    return pd.DataFrame(rows, columns=["mean", "vol"])


def frontier_closed_form(mu, sigma, targets):
    """Shorts-allowed frontier from the textbook A, B, C constants (fast, used for bootstraps).

    var(target) = (C t^2 - 2 B t + A) / (A C - B^2) with A = mu' S^-1 mu, B = 1' S^-1 mu, C = 1' S^-1 1.
    """
    inv = np.linalg.inv(_arr(sigma))
    one, m = np.ones(len(sigma)), _arr(mu)
    A, B, C = m @ inv @ m, one @ inv @ m, one @ inv @ one
    t = _arr(targets)
    var = (C * t**2 - 2 * B * t + A) / (A * C - B**2)
    return pd.DataFrame({"mean": t, "vol": np.sqrt(var)})


def four_portfolios(mu, sigma, rf):
    """GMV and max-Sharpe, with and without shorts, as a weights DataFrame (stocks x portfolio)."""
    idx = list(mu.index) if hasattr(mu, "index") else None
    return pd.DataFrame({
        "GMV (shorts)": gmv(sigma, long_only=False),
        "GMV (long only)": gmv(sigma, long_only=True),
        "Max-Sharpe (shorts)": max_sharpe(mu, sigma, rf, long_only=False),
        "Max-Sharpe (long only)": max_sharpe(mu, sigma, rf, long_only=True),
    }, index=idx)
