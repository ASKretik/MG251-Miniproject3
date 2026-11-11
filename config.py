"""Project settings: the single source of truth for tickers, dates, paths and constants.

Change choices here, not inside src/ or the notebook, so every result traces back to one place.
"""
from pathlib import Path

# --- Universe -------------------------------------------------------------
# 17 US stocks across sectors, all listed since before 2006. Some poor performers
# (C, INTC, F, PFE) are included on purpose to soften "picked today's winners" bias.
SECTORS = {
    "AAPL": "Tech", "MSFT": "Tech", "INTC": "Tech",
    "JPM": "Financials", "C": "Financials",
    "JNJ": "Health", "PFE": "Health",
    "XOM": "Energy",
    "KO": "Staples", "WMT": "Staples",
    "HD": "Discretionary", "F": "Discretionary",
    "CAT": "Industrials", "UNP": "Industrials",
    "NEE": "Utilities",
    "NEM": "Materials",
    "VZ": "Telecom",
}
TICKERS = list(SECTORS)

# Investable market index for comparisons (regressions use French Mkt-RF instead).
MARKET_ETF = "SPY"

# --- Dates (monthly) ------------------------------------------------------
START = "2006-01"      # first month in the sample
END = "2025-12"        # last month; trim if the French library hasn't reached it yet
TRAIN_END = "2015-12"  # train = START..TRAIN_END, test = the month after..END

# Standard sample for the French test portfolios (Phase 6).
FRENCH_START = "1963-07"

# --- French data library files --------------------------------------------
FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"
FF3 = "F-F_Research_Data_Factors"
MOM = "F-F_Momentum_Factor"
FF5 = "F-F_Research_Data_5_Factors_2x3"
PORT_25 = "25_Portfolios_5x5"
PORT_MOM10 = "10_Portfolios_Prior_12_2"

# --- Randomness and resampling --------------------------------------------
SEED = 251
N_BOOT = 500
N_RANDOM_PORTFOLIOS = 20_000

# --- Paths ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "report"

FIG_DPI = 200
