"""Phase 0 sanity checks on config.py."""
import pandas as pd

import config


def test_tickers_match_sectors():
    assert config.TICKERS == list(config.SECTORS)
    assert len(config.TICKERS) == 17


def test_dates_ordered():
    start, train_end, end = (pd.Period(d, "M") for d in (config.START, config.TRAIN_END, config.END))
    assert start < train_end < end
    # Default split: 120 train months, 120 test months.
    assert (train_end - start).n + 1 == 120
    assert (end - train_end).n == 120
