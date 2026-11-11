"""All plotting functions. Every figure is saved to figures/ as a PNG at config.FIG_DPI,
with a title and labelled axes (with units).

Colours: one fixed categorical order (never cycled), a single-hue sequential map for
magnitude, and blue <-> grey <-> red for signed quantities such as alphas.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter, MultipleLocator, NullFormatter, PercentFormatter

import config

# --- Style ----------------------------------------------------------------

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a32a2a"])

# Colour follows the entity: the same strategy is the same colour in every figure.
STRATEGY_COLORS = {
    "GMV (shorts)": SERIES[0],
    "GMV (long only)": SERIES[2],
    "Max-Sharpe (shorts)": SERIES[1],
    "Max-Sharpe (long only)": SERIES[4],
    "Equal weight (1/N)": SERIES[6],
    "SPY": MUTED,
}
ESTIMATOR_COLORS = {
    "Sample": SERIES[1], "CAPM factor": SERIES[3], "FF3 factor": SERIES[0],
    "FF3+Mom factor": SERIES[6], "Ledoit-Wolf": SERIES[2],
}

mpl.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
    "legend.frameon": False, "legend.fontsize": 8.5, "font.size": 9.5,
    "lines.linewidth": 2,
})

PCT = PercentFormatter(1.0, decimals=0)


def save(fig, name):
    """Save to figures/<name>.png at the project dpi and return the path."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    return path


# --- Phase 2 --------------------------------------------------------------

def mean_ci(conf, title="Annualised mean return with 95% confidence interval (train 2006–2015)"):
    order = conf.sort_values("ann_mean").index
    c = conf.loc[order]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(range(len(c)), c["ann_mean"], color=SERIES[0], width=0.6, zorder=2)
    ax.errorbar(range(len(c)), c["ann_mean"],
                yerr=[c["ann_mean"] - c["ci_low"], c["ci_high"] - c["ann_mean"]],
                fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=3)
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set_xticks(range(len(c)), order, rotation=0)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Annualised mean return (% per year)")
    ax.set_xlabel("Stock")
    ax.set_title(title)
    save(fig, "01_mean_ci")
    return fig


# --- Phase 3 --------------------------------------------------------------

def frontier_plot(cloud, front_short, front_long, stocks, points, rf, cal_slope):
    """Everything annualised before plotting: cloud, both frontiers, stocks, key portfolios, CAL."""
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(cloud["vol"], cloud["mean"], c=cloud["sharpe"], cmap=SEQ, s=3, alpha=0.5,
                    lw=0, zorder=1, rasterized=True)
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("Sharpe ratio of random portfolio (annualised)")
    ax.plot(front_short["vol"], front_short["mean"], color=SERIES[1], label="Frontier, shorts allowed", zorder=3)
    ax.plot(front_long["vol"], front_long["mean"], color=SERIES[0], label="Frontier, long only", zorder=3)
    xmax = max(stocks["vol"].max(), front_short["vol"].max()) * 1.05
    xs = np.linspace(0, xmax, 50)
    ax.plot(xs, rf + cal_slope * xs, color=INK2, lw=1.2, ls="--", label="Capital allocation line (shorts)", zorder=2)
    ax.scatter(stocks["vol"], stocks["mean"], color=INK, s=14, zorder=4)
    for t, row in stocks.iterrows():
        ax.annotate(t, (row["vol"], row["mean"]), xytext=(3, 3), textcoords="offset points", fontsize=7.5, color=INK2)
    markers = {"GMV (shorts)": "s", "GMV (long only)": "s", "Max-Sharpe (shorts)": "*", "Max-Sharpe (long only)": "*"}
    for name, (v, m) in points.items():
        ax.scatter(v, m, marker=markers[name], s=170 if "*" == markers[name] else 70,
                   color=STRATEGY_COLORS[name], edgecolor="white", lw=1.5, zorder=5, label=name)
    ax.xaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_xlim(0, xmax)
    ax.set_ylim(min(-0.1, stocks["mean"].min() - 0.03), max(stocks["mean"].max(), front_short["mean"].max()) + 0.03)
    ax.set_xlabel("Annualised volatility (% per year)")
    ax.set_ylabel("Annualised mean return (% per year)")
    ax.set_title("Efficient frontier, train 2006–2015: 20,000 random long-only portfolios vs the optimiser")
    ax.legend(loc="upper left", ncols=2)
    save(fig, "02_frontier")
    return fig


# --- Phase 4 --------------------------------------------------------------

def oos_wealth(port_rets, name="03_oos_wealth", title="Growth of \\$1 out of sample (test 2016–2025, train weights frozen)"):
    wealth = (1 + port_rets).cumprod()
    start = wealth.index[0] - 1
    wealth = pd.concat([pd.DataFrame(1.0, index=[start], columns=wealth.columns), wealth])
    x = wealth.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(9, 5))
    for col in wealth.columns:
        ax.plot(x, wealth[col], color=STRATEGY_COLORS.get(col, INK2), lw=2 if col != "SPY" else 1.5,
                ls="-" if col != "SPY" else "--", label=f"{col}  (\\${wealth[col].iloc[-1]:.2f})")
    ax.set_ylabel(r"Value of \$1 invested (\$)")
    ax.set_xlabel("Month")
    ax.set_title(title)
    ax.legend(loc="upper left", title="Strategy (final value)")
    save(fig, name)
    return fig


def pred_vs_real(table, name="04_pred_vs_real", title="Predicted (train) vs realised (test) Sharpe ratio"):
    t = table.dropna(subset=["pred_sharpe"])
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.2, t["pred_sharpe"], width=0.38, color="#86b6ef", label="Predicted from train data", zorder=2)
    ax.bar(x + 0.2, t["sharpe"], width=0.38, color=SERIES[0], label="Realised on test data", zorder=2)
    for i, (p, r) in enumerate(zip(t["pred_sharpe"], t["sharpe"])):
        ax.text(i - 0.2, p + 0.02, f"{p:.2f}", ha="center", fontsize=8, color=INK2)
        ax.text(i + 0.2, r + 0.02, f"{r:.2f}", ha="center", fontsize=8, color=INK2)
    if "SPY" in table.index:
        ax.axhline(table.loc["SPY", "sharpe"], color=MUTED, ls="--", lw=1.2, label=f"SPY realised ({table.loc['SPY', 'sharpe']:.2f})")
    ax.set_xticks(x, [s.replace(" (", "\n(") for s in t.index])
    ax.set_ylabel("Annualised Sharpe ratio")
    ax.set_xlabel("Strategy")
    ax.set_title(title)
    ax.legend(loc="upper right")
    save(fig, name)
    return fig


# --- Phase 5 --------------------------------------------------------------

def beta_vs_return(beta_tables, mkt_means, fits, name="05_beta_vs_return"):
    """One panel per (label -> (betas, avg excess returns)). CAPM line vs fitted line."""
    fig, axes = plt.subplots(1, len(beta_tables), figsize=(6 * len(beta_tables), 5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (label, (b, r)) in zip(axes, beta_tables.items()):
        ax.scatter(b, r, color=SERIES[0], s=36, edgecolor="white", lw=1, zorder=3)
        for t in b.index:
            ax.annotate(t, (b[t], r[t]), xytext=(3, 3), textcoords="offset points", fontsize=7.5, color=INK2)
        xs = np.linspace(0, b.max() * 1.1, 20)
        ax.plot(xs, mkt_means[label] * xs, color=SERIES[1], label=f"CAPM line (slope {mkt_means[label]:.1%})")
        f = fits[label]
        ax.plot(xs, f["intercept"] + f["slope"] * xs, color=SERIES[6], ls="--",
                label=f"Fitted line (slope {f['slope']:.1%}, R² {f['r2']:.2f})")
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.yaxis.set_major_formatter(PCT)
        ax.set_xlabel("CAPM beta")
        ax.set_title(label, fontsize=10.5)
        ax.legend(loc="upper left")
    axes[0].set_ylabel("Average excess return (% per year)")
    fig.suptitle("Beta vs average excess return for our 17 stocks", fontweight="bold")
    save(fig, name)
    return fig


# --- Phase 6 --------------------------------------------------------------

def actual_vs_predicted(detail, name="06_actual_vs_predicted", subtitle=""):
    fig, axes = plt.subplots(1, len(detail), figsize=(5.2 * len(detail), 5), sharex=True, sharey=True)
    lo = min(min(d[["actual_ann", "predicted_ann"]].min()) for d in detail.values())
    hi = max(max(d[["actual_ann", "predicted_ann"]].max()) for d in detail.values())
    pad = (hi - lo) * 0.08
    for ax, (model, d) in zip(axes, detail.items()):
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#c3c2b7", lw=1.2, zorder=1, label="45° line (zero alpha)")
        is_mom = d.index.str.startswith("M")
        ax.scatter(d.loc[~is_mom, "predicted_ann"], d.loc[~is_mom, "actual_ann"], color=SERIES[0], s=30,
                   edgecolor="white", lw=1, zorder=3, label="25 size × B/M")
        ax.scatter(d.loc[is_mom, "predicted_ann"], d.loc[is_mom, "actual_ann"], color=SERIES[1], marker="D", s=26,
                   edgecolor="white", lw=1, zorder=3, label="10 momentum")
        worst = d["alpha_ann"].abs().nlargest(6).index
        for p in worst:
            ax.annotate(p, (d.loc[p, "predicted_ann"], d.loc[p, "actual_ann"]), xytext=(4, -3),
                        textcoords="offset points", fontsize=7.5, color=INK2)
        ax.set_title(f"{model}: mean |α| {d['alpha_ann'].abs().mean():.1%}", fontsize=10.5)
        ax.set_xlabel("Predicted avg excess return (% per year)")
        ax.xaxis.set_major_locator(MultipleLocator(0.04))
        ax.yaxis.set_major_locator(MultipleLocator(0.04))
        ax.xaxis.set_major_formatter(PCT)
        ax.yaxis.set_major_formatter(PCT)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
    axes[0].set_ylabel("Actual avg excess return (% per year)")
    axes[0].legend(loc="upper left")
    fig.suptitle(f"Actual vs model-predicted average returns, French portfolios {subtitle}", fontweight="bold")
    save(fig, name)
    return fig


def alpha_heatmap(details, name="07_alpha_heatmap", subtitle=""):
    """5x5 grid of annualised alphas (size rows, B/M columns) for each model in `details`."""
    grids = {m: d.loc[[f"S{i}B{j}" for i in range(1, 6) for j in range(1, 6)], "alpha_ann"].to_numpy().reshape(5, 5)
             for m, d in details.items()}
    vmax = max(np.abs(g).max() for g in grids.values())
    norm = TwoSlopeNorm(0, -vmax, vmax)
    fig, axes = plt.subplots(1, len(grids), figsize=(5.6 * len(grids), 4.8), gridspec_kw={"wspace": 0.35})
    for ax, (m, g) in zip(axes, grids.items()):
        im = ax.imshow(g, cmap=DIV, norm=norm)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{g[i, j]:.1%}", ha="center", va="center", fontsize=8,
                        color="white" if abs(g[i, j]) > 0.6 * vmax else INK)
        ax.set_xticks(range(5), ["B1\nGrowth", "B2", "B3", "B4", "B5\nValue"])
        ax.set_yticks(range(5), ["S1 Small", "S2", "S3", "S4", "S5 Big"])
        ax.set_xlabel("Book-to-market quintile")
        ax.set_ylabel("Size quintile")
        ax.grid(False)
        ax.set_title(f"{m} alpha", fontsize=10.5)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, format=PercentFormatter(1.0, decimals=0))
    cb.set_label("Annualised alpha (% per year)")
    fig.suptitle(f"Pricing errors on the 25 size × book-to-market portfolios {subtitle}", fontweight="bold")
    save(fig, name)
    return fig


# --- Phase 7 --------------------------------------------------------------

def weight_stability(boot_weights, name="08_weight_stability"):
    """Box plots of bootstrapped GMV weights per stock, one box per estimator side by side."""
    labels = list(boot_weights)
    tickers = boot_weights[labels[0]].columns
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.8 / len(labels)
    for k, lab in enumerate(labels):
        pos = np.arange(len(tickers)) + (k - (len(labels) - 1) / 2) * width
        col = ESTIMATOR_COLORS.get(lab, SERIES[k])
        ax.boxplot([boot_weights[lab][t] for t in tickers], positions=pos, widths=width * 0.85,
                   patch_artist=True, showfliers=False, whis=(2.5, 97.5),
                   boxprops=dict(facecolor=col, edgecolor=col, alpha=0.75),
                   medianprops=dict(color="white", lw=1.5), whiskerprops=dict(color=col),
                   capprops=dict(color=col))
        ax.plot([], [], color=col, lw=6, label=f"{lab} Σ (avg 95% range {boot_weights[lab].quantile(0.975).sub(boot_weights[lab].quantile(0.025)).mean():.0%})")
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set_xticks(range(len(tickers)), tickers)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_xlabel("Stock")
    ax.set_ylabel("GMV weight (shorts allowed, % of portfolio)")
    ax.set_title(f"GMV weights across {config.N_BOOT} bootstrap resamples of the train months (box = IQR, whiskers = 95%)")
    ax.legend(loc="upper left")
    save(fig, name)
    return fig


def rolling_weights_plot(paths, name="09_rolling_gmv_weights"):
    """Rolling 60-month GMV weights over time, one panel per estimator, same y-scale."""
    fig, axes = plt.subplots(1, len(paths), figsize=(6 * len(paths), 4.5), sharey=True)
    for ax, (lab, (W, to)) in zip(axes, paths.items()):
        x = W.index.to_timestamp()
        for t in W.columns:
            ax.plot(x, W[t], lw=1, color=SEQ(0.35 + 0.6 * (list(W.columns).index(t) / len(W.columns))), alpha=0.9)
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.set_title(f"{lab} Σ: average turnover {to:.0%} per month", fontsize=10.5)
        ax.set_xlabel("Month weights are held")
        ax.yaxis.set_major_formatter(PCT)
    axes[0].set_ylabel("GMV weight (% of portfolio)")
    fig.suptitle("Rolling 60-month GMV weights (shorts allowed), one line per stock", fontweight="bold")
    save(fig, name)
    return fig


def gmv_vol_by_estimator(table, name="10_gmv_vol_by_estimator"):
    """Horizontal bars of realised GMV volatility per covariance estimator (lower is better)."""
    t = table.sort_values()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh(range(len(t)), t, color=[ESTIMATOR_COLORS.get(i, SERIES[0]) for i in t.index], height=0.6, zorder=2)
    for i, v in enumerate(t):
        ax.text(v + 0.001, i, f"{v:.1%}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(t)), t.index)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlim(t.min() * 0.85, t.max() * 1.06)
    ax.set_xlabel("Realised annualised volatility of rolling GMV (% per year, lower is better)")
    ax.set_title("Out-of-sample risk of the rolling 60-month GMV portfolio by covariance estimator")
    save(fig, name)
    return fig


# --- Phase 8: original visualisations -------------------------------------

def diversification_curve(curve, sigma_bar, rhos, name="11_diversification"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ns = curve.index.to_numpy()
    for k, rho in enumerate(rhos):
        theory = sigma_bar * np.sqrt(rho + (1 - rho) / ns)
        ax.plot(ns, theory, color=[SERIES[3], SERIES[2], SERIES[6]][k], lw=1.5, ls="--",
                label=f"Theory, ρ = {rho}: floor σ√ρ = {sigma_bar * np.sqrt(rho):.0%}")
    ax.plot(ns, curve["mean"], color=SERIES[0], marker="o", ms=5, label="Our stocks (average of random subsets)")
    ax.fill_between(ns, curve["p10"], curve["p90"], color=SERIES[0], alpha=0.15, lw=0, label="10th–90th percentile of subsets")
    ax.yaxis.set_major_formatter(PCT)
    ax.set_xticks(ns)
    ax.set_ylim(0, None)
    ax.set_xlabel("Number of stocks in equal-weight portfolio")
    ax.set_ylabel("Annualised volatility (% per year)")
    ax.set_title("Diversification: risk falls fast, then hits the market floor (2006–2025)")
    ax.legend(loc="upper right")
    save(fig, name)
    return fig


def wobbling_frontier(fronts, gmv_pts, ms_pts, base_front, name="12_wobbling_frontier"):
    fig, ax = plt.subplots(figsize=(9, 6))
    for f in fronts:
        ax.plot(f["vol"], f["mean"], color=SERIES[0], lw=0.7, alpha=0.25)
    ax.plot(base_front["vol"], base_front["mean"], color=INK, lw=2, label="Frontier from the actual train data")
    ax.plot([], [], color=SERIES[0], lw=1, label=f"{len(fronts)} bootstrap frontiers")
    ax.scatter(gmv_pts["vol"], gmv_pts["mean"], color=SERIES[2], s=12, alpha=0.7, label="GMV, each bootstrap", zorder=3)
    ax.scatter(ms_pts["vol"], ms_pts["mean"], color=SERIES[1], s=12, alpha=0.7, label="Max-Sharpe, each bootstrap", zorder=3)
    ax.xaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_xlim(0, 0.45)
    ax.set_ylim(-0.05, 0.6)
    ax.set_xlabel("Annualised volatility (% per year)")
    ax.set_ylabel("Annualised mean return (% per year)")
    ax.set_title("The wobbling frontier: same stocks, resampled history (long-only, train)")
    ax.legend(loc="upper left")
    save(fig, name)
    return fig


def error_maximiser(base, bumped_mu, bumped_vol, stock, name="13_error_maximiser"):
    tickers = base.index
    x = np.arange(len(tickers))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(x - 0.27, base, width=0.26, color="#9ec5f4", label="Max-Sharpe weights, original inputs", zorder=2)
    ax.bar(x, bumped_mu, width=0.26, color=SERIES[1], label=f"{stock} mean +1 standard error", zorder=2)
    ax.bar(x + 0.27, bumped_vol, width=0.26, color=SERIES[2], label=f"{stock} volatility +1 standard error", zorder=2)
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set_xticks(x, tickers)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_xlabel("Stock")
    ax.set_ylabel("Portfolio weight (% of portfolio)")
    tm = np.abs(bumped_mu - base).sum()
    tv = np.abs(bumped_vol - base).sum()
    ax.set_title(f"Error maximiser: a 1-SE nudge to {stock}'s mean moves {tm:.0%} of the portfolio; to its vol, {tv:.0%}")
    ax.legend(loc="upper left")
    save(fig, name)
    return fig


def fund_alpha(alphas, tstats, fund, name="14_fund_alpha"):
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    x = np.arange(len(alphas))
    bars = ax.bar(x, alphas, color=[SERIES[1], SERIES[0], SERIES[6]][: len(alphas)], width=0.55, zorder=2)
    for i, (a, t) in enumerate(zip(alphas, tstats)):
        ax.text(i, a + (0.002 if a >= 0 else -0.004), f"{a:.1%}\n(t = {t:.1f})", ha="center",
                va="bottom" if a >= 0 else "top", fontsize=9, color=INK2)
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set_xticks(x, alphas.index)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax.set_xlabel("Model")
    ax.set_ylabel("Annualised alpha (% per year)")
    ax.set_title(f"{fund}: the alpha shrinks as factors are added (2006–2025)")
    pad = max(abs(alphas).max() * 0.6, 0.01)
    ax.set_ylim(min(0, alphas.min()) - pad, max(0, alphas.max()) + pad)
    save(fig, name)
    return fig


def factor_correlation(corr, rolling, name="15_factor_correlation"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.25]})
    im = ax1.imshow(corr, cmap=DIV, vmin=-1, vmax=1)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.iloc[i, j]
            ax1.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5, color="white" if abs(v) > 0.6 else INK)
    ax1.set_xticks(range(len(corr)), corr.columns, rotation=0)
    ax1.set_yticks(range(len(corr)), corr.index)
    ax1.grid(False)
    ax1.set_title("Full sample, Jul 1963 – latest", fontsize=10.5)
    fig.colorbar(im, ax=ax1, shrink=0.8).set_label("Correlation")
    x = rolling.index.to_timestamp()
    for k, col in enumerate(rolling.columns):
        ax2.plot(x, rolling[col], color=SERIES[k], lw=1.8, label=col)
    ax2.axhline(0, color="#c3c2b7", lw=1)
    ax2.set_ylim(-1, 1)
    ax2.set_xlabel("End of 10-year window")
    ax2.set_ylabel("Rolling 120-month correlation")
    ax2.set_title("Rolling 10-year correlations of selected pairs", fontsize=10.5)
    ax2.legend(loc="lower left")
    fig.suptitle("How related are the factors? Fama–French 5 + momentum", fontweight="bold")
    save(fig, name)
    return fig


def survivorship(wealth, name="16_survivorship"):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = wealth.index.to_timestamp()
    cols = [SERIES[1], SERIES[0], MUTED]
    for k, col in enumerate(wealth.columns):
        ax.plot(x, wealth[col], color=cols[k], lw=2 if k < 2 else 1.5, ls="-" if k < 2 else "--",
                label=f"{col}  (\\${wealth[col].iloc[-1]:.1f})")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"\\${v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Month")
    ax.set_ylabel(r"Value of \$1 invested, log scale (\$)")
    ax.set_title("Survivorship bias: picking the winners you already know about (2006–2025)")
    ax.legend(loc="upper left", title="Equal-weight portfolio (final value)")
    save(fig, name)
    return fig


def smb_no_premium(r2_table, cum_smb, split, name="17_smb_movement_no_premium"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(r2_table))
    ax1.bar(x - 0.2, r2_table["CAPM"], width=0.38, color="#86b6ef", label="Market only", zorder=2)
    ax1.bar(x + 0.2, r2_table["Mkt + SMB"], width=0.38, color=SERIES[0], label="Market + SMB", zorder=2)
    ax1.set_xticks(x, r2_table.index)
    ax1.yaxis.set_major_formatter(PCT)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("Size quintile (equal mix of the 5 B/M portfolios)")
    ax1.set_ylabel("Time-series R² (% of variance explained)")
    ax1.set_title("SMB explains a lot of small-stock movement…", fontsize=10.5)
    ax1.legend(loc="lower right")
    xt = cum_smb.index.to_timestamp()
    ax2.plot(xt, cum_smb, color=SERIES[0], lw=1.6)
    ax2.axvline(pd.Period(split, "M").to_timestamp(), color=SERIES[1], ls="--", lw=1.2,
                label=f"Banz (1981) publishes the size effect")
    ax2.set_yscale("log")
    ax2.yaxis.set_major_locator(mpl.ticker.FixedLocator([0.5, 1, 2, 3, 4, 6]))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"\\${v:g}"))
    ax2.yaxis.set_minor_formatter(NullFormatter())
    ax2.set_xlabel("Month")
    ax2.set_ylabel(r"Value of \$1 in SMB long-short, log scale (\$)")
    ax2.set_title("…but has earned little since it was published", fontsize=10.5)
    ax2.legend(loc="upper left")
    fig.suptitle("A factor that moves returns without paying a premium: SMB", fontweight="bold")
    save(fig, name)
    return fig
