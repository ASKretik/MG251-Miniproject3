"""Phase 1 sanity checks: French CSV parsing and the processed monthly dataset."""
import numpy as np
import pandas as pd
import pytest

import config
from src import data

FAKE_FRENCH = """This file was created by CMPT_ME_BEME_RETS using the 202509 CRSP database.
Some free-text header

,Mkt-RF,SMB,HML,RF
192607,    2.89,   -2.55,   -2.39,    0.22
192608,    2.64,   -1.14,    3.81,    0.25
192609,  -99.99,   -1.37,    0.05,    0.23

 Annual Factors: January-December
,Mkt-RF,SMB,HML,RF
1927,   29.47,   -2.46,   -3.75,    3.12
"""


def test_parse_french_monthly_block_only():
    df = data.parse_french_csv(FAKE_FRENCH)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert len(df) == 3                               # annual block ignored
    assert df.index.freqstr.startswith("M")
    assert df.index[0] == pd.Period("1926-07", "M")
    assert np.isclose(df.loc["1926-07", "Mkt-RF"], 0.0289)  # percent -> decimal
    assert np.isnan(df.loc["1926-09", "Mkt-RF"])       # -99.99 -> NaN


@pytest.fixture(scope="module")
def monthly():
    path = config.PROCESSED_DIR / "monthly.csv"
    if not path.exists():
        pytest.skip("run the notebook (or data.build_monthly_dataset()) first")
    return data.load_monthly_dataset()


def test_no_missing_months(monthly):
    checks = data.sanity_checks(monthly)
    assert checks["n_months"] == 240
    assert checks["no_missing_months"]
    assert checks["no_missing_values"]


def test_spy_tracks_french_market(monthly):
    assert data.sanity_checks(monthly)["corr_spy_mkt"] > 0.98


def test_market_premium_plausible(monthly):
    assert 0.05 < data.sanity_checks(monthly)["mkt_rf_ann_mean"] < 0.12


def test_returns_are_decimals(monthly):
    # Monthly returns in decimals: the typical absolute return is a few percent, not a few units.
    assert monthly[config.TICKERS].abs().median().max() < 0.2
    assert monthly["RF"].max() < 0.01
