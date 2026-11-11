"""Estimation of means and covariances, their standard errors, and bootstrap intervals.

All estimates used for choosing weights come from train data only.
"""
import numpy as np
import pandas as pd

import config


def split_train_test(df, train_end=config.TRAIN_END):
    """Split at `train_end` (inclusive). Asserts the halves don't overlap (no look-ahead)."""
    cut = pd.Period(train_end, "M")
    train, test = df.loc[:cut], df.loc[cut + 1:]
    assert train.index.max() < test.index.min(), "train and test overlap"
    return train, test


def excess_returns(df, cols=config.TICKERS):
    """Returns minus the 1-month T-bill rate, month by month."""
    return df[cols].sub(df["RF"], axis=0)


def mean_cov(returns):
    """Sample mean vector and covariance matrix (monthly, ddof=1)."""
    return returns.mean(), returns.cov()


def mean_confidence(returns, z=1.96):
    """Annualised mean with its standard error (sigma / sqrt(T)) and a 95% CI.

    The SE shrinks with sqrt(T) in *months of history*, not with how often we sample.
    """
    T = len(returns)
    mean_m, sd_m = returns.mean(), returns.std()
    se_m = sd_m / np.sqrt(T)
    return pd.DataFrame({
        "ann_mean": mean_m * 12,
        "ann_se": se_m * 12,
        "ci_low": (mean_m - z * se_m) * 12,
        "ci_high": (mean_m + z * se_m) * 12,
    })


def bootstrap_months(returns, n_boot=config.N_BOOT, seed=config.SEED):
    """Yield bootstrap samples made by drawing whole months with replacement.

    Drawing whole months keeps the cross-sectional correlation inside each month intact.
    """
    rng = np.random.default_rng(seed)
    values = returns.to_numpy()
    T = len(values)
    for _ in range(n_boot):
        idx = rng.integers(0, T, size=T)
        yield pd.DataFrame(values[idx], columns=returns.columns)


def bootstrap_vol_corr(returns, pairs, n_boot=config.N_BOOT, seed=config.SEED):
    """Bootstrap 95% intervals for annualised vols and for the correlations in `pairs`.

    Returns (vol_table, corr_table), each with point estimate, bootstrap SE and 2.5/97.5% bounds.
    """
    vols, corrs = [], []
    for sample in bootstrap_months(returns, n_boot, seed):
        vols.append(sample.std() * np.sqrt(12))
        c = sample.corr()
        corrs.append(pd.Series({f"{a}-{b}": c.loc[a, b] for a, b in pairs}))
    vols, corrs = pd.DataFrame(vols), pd.DataFrame(corrs)

    point_vol = returns.std() * np.sqrt(12)
    c0 = returns.corr()
    point_corr = pd.Series({f"{a}-{b}": c0.loc[a, b] for a, b in pairs})

    def table(point, draws):
        return pd.DataFrame({
            "estimate": point,
            "boot_se": draws.std(),
            "ci_low": draws.quantile(0.025),
            "ci_high": draws.quantile(0.975),
        })

    return table(point_vol, vols), table(point_corr, corrs)


def relative_precision(returns, vol_table):
    """Relative SE of the mean (SE / |mean|) next to relative SE of the vol (boot SE / vol).

    The comparison shows why means are the weak link: the vol is pinned down to a few percent
    of its value, the mean often to no better than +/- its own size.
    """
    conf = mean_confidence(returns)
    return pd.DataFrame({
        "ann_mean": conf["ann_mean"],
        "rel_se_mean": conf["ann_se"] / conf["ann_mean"].abs(),
        "ann_vol": vol_table["estimate"],
        "rel_se_vol": vol_table["boot_se"] / vol_table["estimate"],
    })


def n_parameters(n):
    """Number of inputs a mean-variance optimiser needs: N means + N(N+1)/2 covariance terms."""
    return {"means": n, "cov_terms": n * (n + 1) // 2, "total": n + n * (n + 1) // 2}


def years_needed(ann_vol, half_width, z=1.96):
    """Years of monthly data so the 95% CI of the annualised mean is +/- `half_width`.

    CI half-width = z * ann_vol / sqrt(years), so years = (z * ann_vol / half_width)^2.
    """
    return (z * ann_vol / half_width) ** 2


def covariance_estimators(train, tickers=config.TICKERS):
    """Every covariance estimator compared in Phase 7, all fitted on `train` only.

    - Sample: plain sample covariance (N(N+1)/2 free numbers).
    - CAPM / FF3 / FF3+Mom factor: B Sigma_F B' + D from regressions on train.
    - Ledoit-Wolf: sample covariance shrunk towards a scaled identity.
    """
    from sklearn.covariance import LedoitWolf

    from src import factors as fm

    ex = excess_returns(train, tickers)
    out = {"Sample": train[tickers].cov()}
    for name, cols in fm.MODELS.items():
        out[f"{name} factor"] = fm.factor_covariance(ex, train[cols])[0]
    lw = LedoitWolf().fit(train[tickers].to_numpy())
    # sklearn uses divisor T; rescale to T-1 so it is on the same footing as .cov()
    T = len(train)
    out["Ledoit-Wolf"] = pd.DataFrame(lw.covariance_ * T / (T - 1), index=tickers, columns=tickers)
    return out
