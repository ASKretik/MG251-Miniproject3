"""Data loading: yfinance prices and the Ken French data library.

Conventions (see plan.md):
- returns are monthly simple returns in decimals, indexed by pd.Period("M");
- French files come in percent and are divided by 100 on load;
- raw downloads are cached in data/raw/ but everything can be rebuilt from an empty data/.
"""
import io
import zipfile

import numpy as np
import pandas as pd
import requests

import config


# --- Prices ---------------------------------------------------------------

def load_prices(tickers, start, end, cache=True, cache_name="prices_daily"):
    """Month-end adjusted close prices for `tickers`, indexed by monthly Period.

    Adjusted close includes splits and dividends, so its pct_change is a total return.
    We download daily data and keep the last price in each month.
    """
    import yfinance as yf

    path = config.RAW_DIR / f"{cache_name}.csv"
    if cache and path.exists():
        daily = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        # Start one month early so the first sample month has a return.
        first = (pd.Period(start, "M") - 1).start_time.strftime("%Y-%m-%d")
        last = (pd.Period(end, "M") + 1).start_time.strftime("%Y-%m-%d")
        raw = yf.download(list(tickers), start=first, end=last, auto_adjust=True,
                          progress=False, threads=False)
        daily = raw["Close"]
        if isinstance(daily, pd.Series):
            daily = daily.to_frame(tickers[0])
        missing = [t for t in tickers if t not in daily or daily[t].dropna().empty]
        if missing:
            raise RuntimeError(f"yfinance returned no data for {missing}; stop and check tickers.")
        config.RAW_DIR.mkdir(parents=True, exist_ok=True)
        daily.to_csv(path)

    monthly = daily[list(tickers)].resample("ME").last()
    monthly.index = monthly.index.to_period("M")
    return monthly


def load_returns(tickers, start, end, cache_name, cache=True):
    """Monthly returns for any extra tickers (Phase 8 experiments), cached under `cache_name`.

    Tickers that listed after `start` simply have NaN before their first month.
    """
    return prices_to_returns(load_prices(tickers, start, end, cache, cache_name), start, end)


def prices_to_returns(prices, start, end):
    """Monthly simple returns, trimmed to [start, end]."""
    rets = prices.pct_change(fill_method=None)
    return rets.loc[pd.Period(start, "M"):pd.Period(end, "M")]


# --- Ken French data library ----------------------------------------------

def _download_french_csv(name, cache=True):
    """Return the text of the CSV inside the French library zip `name`, cached in data/raw/."""
    path = config.RAW_DIR / f"{name}.csv"
    if cache and path.exists():
        return path.read_text(encoding="latin-1")
    url = config.FRENCH_URL.format(name=name)
    resp = requests.get(url, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"French library returned {resp.status_code} for {url}; "
                           "check the data library page for the current file name.")
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        text = zf.read(zf.namelist()[0]).decode("latin-1")
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="latin-1")
    return text


def parse_french_csv(text):
    """Parse only the first monthly block of a French library CSV.

    The files have free-text headers, then a header row starting with a comma, then rows
    "YYYYMM, x, y, ...", then a blank line before the next block (annual, or equal-weighted).
    We take the first block whose dates are 6-digit YYYYMM (monthly, value-weighted).
    """
    lines = text.splitlines()
    header, rows = None, []
    for line in lines:
        cells = [c.strip() for c in line.split(",")]
        first = cells[0]
        if header is None:
            # Header row: empty first cell followed by column names.
            if first == "" and len(cells) > 1 and any(cells[1:]):
                header = cells[1:]
            continue
        if first.isdigit() and len(first) == 6:
            rows.append(cells[: len(header) + 1])
        elif rows:
            break  # first non-data line after the monthly block ends it
        else:
            header = None  # header belonged to a block without monthly rows; keep looking
    if not rows:
        raise ValueError("No monthly block found in French CSV.")

    df = pd.DataFrame([r[1:] for r in rows], columns=header,
                      index=pd.PeriodIndex([r[0] for r in rows], freq="M"))
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([-99.99, -999.0], np.nan)
    df.index.name = "month"
    return df / 100.0  # percent -> decimal


def load_french(name, cache=True):
    """Monthly (value-weighted) table from French library file `name`, in decimals."""
    return parse_french_csv(_download_french_csv(name, cache=cache))


# --- One tidy dataset -----------------------------------------------------

def build_monthly_dataset(start=config.START, end=config.END, cache=True):
    """Stock returns, SPY return and French factors aligned on common months.

    Columns: the 17 tickers, SPY, Mkt-RF, SMB, HML, RF, Mom. Saved to data/processed/monthly.csv.
    """
    tickers = config.TICKERS + [config.MARKET_ETF]
    rets = prices_to_returns(load_prices(tickers, start, end, cache=cache), start, end)

    ff3 = load_french(config.FF3, cache=cache)
    mom = load_french(config.MOM, cache=cache)
    mom.columns = [c.strip() for c in mom.columns]
    factors = ff3.join(mom[["Mom"]], how="inner")

    df = rets.join(factors, how="inner")
    df = df.loc[pd.Period(start, "M"):pd.Period(end, "M")]
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.PROCESSED_DIR / "monthly.csv")
    return df


def load_monthly_dataset():
    """Read the processed dataset back with a monthly PeriodIndex."""
    df = pd.read_csv(config.PROCESSED_DIR / "monthly.csv", index_col=0)
    df.index = pd.PeriodIndex(df.index, freq="M")
    return df


# --- Sanity checks --------------------------------------------------------

def sanity_checks(df, tickers=config.TICKERS):
    """Run the Phase 1 checks and return a dict of results (also used by tests)."""
    expected = pd.period_range(df.index.min(), df.index.max(), freq="M")
    out = {
        "n_months": len(df),
        "no_missing_months": len(df.index) == len(expected) and (df.index == expected).all(),
        "no_missing_values": not df[tickers + [config.MARKET_ETF]].isna().any().any(),
    }
    stacked = df[tickers].stack()
    out["extreme_returns"] = stacked[stacked.abs() > 0.5]
    out["corr_spy_mkt"] = df[config.MARKET_ETF].corr(df["Mkt-RF"] + df["RF"])
    out["mkt_rf_ann_mean"] = df["Mkt-RF"].mean() * 12
    return out


def summary_table(df, tickers=config.TICKERS):
    """Per-stock annualised mean and vol, plus worst and best month."""
    r = df[tickers]
    return pd.DataFrame({
        "sector": [config.SECTORS.get(t, "") for t in tickers],
        "ann_mean": r.mean() * 12,
        "ann_vol": r.std() * np.sqrt(12),
        "min_month": r.min(),
        "max_month": r.max(),
    })
