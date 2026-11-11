"""Factor models: time-series regressions, cross-sectional CAPM checks, the GRS test,
and factor-model covariance matrices (Sigma = B Sigma_F B' + D).

All regressions use *excess* returns (return minus RF) on the left-hand side.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

CAPM = ["Mkt-RF"]
FF3 = ["Mkt-RF", "SMB", "HML"]
CARHART = ["Mkt-RF", "SMB", "HML", "Mom"]
MODELS = {"CAPM": CAPM, "FF3": FF3, "FF3+Mom": CARHART}


# --- Time-series regressions on individual stocks (Phase 5) ---------------

def ts_regression(y, X, nw_lags=6):
    """OLS of y on a constant and X with Newey-West (HAC) standard errors.

    Returns the fitted statsmodels result. HAC SEs allow for heteroskedastic and
    autocorrelated errors, which monthly stock returns have (volatility clusters).
    """
    Xc = sm.add_constant(X)
    return sm.OLS(y, Xc, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})


def capm_table(excess, mkt, nw_lags=6):
    """Per-stock CAPM alpha (annualised), beta, Newey-West t-stats and R^2.

    Also computes beta by hand as cov(r_i, m) / var(m) and asserts it matches OLS.
    """
    rows = {}
    for t in excess.columns:
        res = ts_regression(excess[t], mkt, nw_lags)
        beta_hand = excess[t].cov(mkt) / mkt.var()
        assert np.isclose(beta_hand, res.params[mkt.name]), f"beta mismatch for {t}"
        rows[t] = {
            "alpha_ann": res.params["const"] * 12,
            "t_alpha": res.tvalues["const"],
            "beta": res.params[mkt.name],
            "t_beta": res.tvalues[mkt.name],
            "r2": res.rsquared,
        }
    return pd.DataFrame(rows).T


def cross_section(betas, avg_excess):
    """Regress average excess return on beta across stocks (one observation per stock).

    Returns dict with fitted intercept, slope and R^2. Compare the slope with the CAPM slope,
    which is the average market excess return.
    """
    res = sm.OLS(avg_excess, sm.add_constant(betas)).fit()
    return {"intercept": res.params.iloc[0], "slope": res.params.iloc[1], "r2": res.rsquared}


# --- Many test assets at once (Phase 6) -----------------------------------

def multi_regression(excess, factors):
    """OLS of every column of `excess` (T x N) on a constant and `factors` (T x L).

    Plain least squares in matrix form so residuals are easy to reuse in the GRS test.
    Returns dict: alpha (N), betas (N x L), resid (T x N), r2 (N).
    """
    Y = excess.to_numpy()
    F = factors.to_numpy()
    X = np.column_stack([np.ones(len(F)), F])
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ coef
    r2 = 1 - resid.var(axis=0) / Y.var(axis=0)
    cols = excess.columns
    return {
        "alpha": pd.Series(coef[0], index=cols),
        "betas": pd.DataFrame(coef[1:].T, index=cols, columns=factors.columns),
        "resid": pd.DataFrame(resid, index=excess.index, columns=cols),
        "r2": pd.Series(r2, index=cols),
    }


def grs_test(alpha, resid, factors):
    """Gibbons-Ross-Shanken test that all N intercepts are jointly zero.

    GRS = ((T - N - L) / N) * a' S^-1 a / (1 + f' O^-1 f)  ~  F(N, T - N - L)

    a : (N,) estimated alphas (monthly) from the time-series regressions
    S : (N x N) residual covariance, e'e / T  (how noisy each alpha is, and how they co-move)
    f : (L,) mean factor returns
    O : (L x L) factor covariance with divisor T
    T : months, N : test assets, L : factors

    The numerator a' S^-1 a is a squared, risk-adjusted size of the pricing errors. The
    denominator (1 + squared Sharpe of the factors) corrects for alphas being estimated
    with error that grows when the factors themselves are volatile relative to their mean.
    H0: every alpha is zero, i.e. the factors fully explain average returns.
    """
    a = np.asarray(alpha, dtype=float)
    E = np.asarray(resid, dtype=float)
    F = np.asarray(factors, dtype=float)
    T, N = E.shape
    L = F.shape[1]
    S = E.T @ E / T
    f_bar = F.mean(axis=0)
    Fd = F - f_bar
    O = Fd.T @ Fd / T
    stat = (T - N - L) / N * (a @ np.linalg.solve(S, a)) / (1 + f_bar @ np.linalg.solve(O, f_bar))
    p = stats.f.sf(stat, N, T - N - L)
    return stat, p


def price_test_assets(excess, factor_df, models=MODELS):
    """Run every model on the test assets. Returns (summary table, per-asset detail dict).

    predicted average excess return = betas @ mean factor returns; actual = sample average;
    pricing error (alpha) = actual - predicted.
    """
    summary, detail = {}, {}
    for name, cols in models.items():
        F = factor_df[cols]
        reg = multi_regression(excess, F)
        predicted = reg["betas"] @ F.mean()
        actual = excess.mean()
        stat, p = grs_test(reg["alpha"], reg["resid"], F)
        a_ann = reg["alpha"] * 12
        summary[name] = {
            "mean_abs_alpha_ann": a_ann.abs().mean(),
            "rms_alpha_ann": np.sqrt((a_ann**2).mean()),
            "avg_r2": reg["r2"].mean(),
            "GRS": stat,
            "GRS_p": p,
        }
        detail[name] = pd.DataFrame({
            "actual_ann": actual * 12,
            "predicted_ann": predicted * 12,
            "alpha_ann": a_ann,
            "r2": reg["r2"],
        })
    return pd.DataFrame(summary).T, detail


def rename_25(cols):
    """'SMALL LoBM', 'ME1 BM2', ... -> 'S1B1', 'S1B2', ... (row-major: size outer, B/M inner)."""
    return [f"S{i // 5 + 1}B{i % 5 + 1}" for i in range(len(cols))]


def rename_mom10(cols):
    """'Lo PRIOR', 'PRIOR 2', ..., 'Hi PRIOR' -> 'M1' (losers) ... 'M10' (winners)."""
    return [f"M{i + 1}" for i in range(len(cols))]


# --- Factor-model covariance (Phase 7) ------------------------------------

def factor_covariance(excess, factors):
    """Sigma = B Sigma_F B' + D from a time-series regression of each stock on the factors.

    B (N x K) loadings, Sigma_F (K x K) sample factor covariance, D diagonal residual variances.
    The key assumption is that residuals are uncorrelated across stocks; all co-movement
    runs through the factors. Returns (Sigma DataFrame, dict of the pieces).
    """
    reg = multi_regression(excess, factors)
    B = reg["betas"].to_numpy()
    sigma_f = factors.cov().to_numpy()
    d = reg["resid"].var(ddof=len(factors.columns) + 1).to_numpy()
    sigma = B @ sigma_f @ B.T + np.diag(d)
    cols = excess.columns
    return pd.DataFrame(sigma, index=cols, columns=cols), {
        "B": reg["betas"], "sigma_f": sigma_f, "D": pd.Series(d, index=cols), "resid": reg["resid"],
    }


def factor_implied_mu(excess, factors):
    """Expected excess returns implied by the model: B @ mean factor returns (no alpha)."""
    reg = multi_regression(excess, factors)
    return reg["betas"] @ factors.mean()


def residual_correlation(resid, sectors=None):
    """Average off-diagonal residual correlation, plus the most-correlated pairs.

    If the factor model's assumption holds, these should be near zero.
    """
    c = resid.corr()
    n = len(c)
    mask = ~np.eye(n, dtype=bool)
    avg = c.where(mask).stack().mean()
    pairs = (c.where(np.triu(np.ones((n, n), dtype=bool), 1)).stack()
             .sort_values(key=np.abs, ascending=False).rename("resid_corr").to_frame())
    if sectors:
        pairs["same_sector"] = [sectors[a] == sectors[b] for a, b in pairs.index]
    return avg, pairs


def parameter_counts(ns=(17, 20, 500), ks=(1, 3, 4)):
    """Covariance parameters: sample N(N+1)/2 vs factor model NK + K(K+1)/2 + N."""
    rows = []
    for n in ns:
        row = {"N": n, "sample": n * (n + 1) // 2}
        for k in ks:
            row[f"factor K={k}"] = n * k + k * (k + 1) // 2 + n
        rows.append(row)
    return pd.DataFrame(rows).set_index("N")
