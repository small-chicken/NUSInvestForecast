"""Loading, cleaning, and aligning raw price data from data/raw/Delta1/."""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "Delta1"
FUTURES_DIR = RAW_DIR / "Futures Data"
ETF_DIR = RAW_DIR / "ETF Data"


def load_futures_catalogue() -> pd.DataFrame:
    """Metadata for every futures symbol (name, currency, class, tick size, ...)."""
    return pd.read_csv(RAW_DIR / "CATALOGUE_Delta1_Futures.csv")


def load_etf_catalogue() -> pd.DataFrame:
    """Metadata for every ETF symbol (name, exchange, business summary, ...)."""
    return pd.read_csv(RAW_DIR / "CATALOGUE_Delta1_ETF.csv")


def load_futures(symbol: str) -> pd.DataFrame:
    """OHLCV + open interest for one futures symbol, indexed by Date.

    `symbol` should include the leading `&` (raw continuous contract) or
    `&..._CCB` (back-adjusted continuous contract), matching the catalogue.
    """
    df = pd.read_csv(FUTURES_DIR / f"{symbol}.csv", parse_dates=["Date"])
    return df.set_index("Date").sort_index()


def load_etf(symbol: str) -> pd.DataFrame:
    """OHLCV + dividends for one ETF symbol, indexed by Date."""
    df = pd.read_csv(ETF_DIR / f"{symbol}.csv", parse_dates=["Date"])
    return df.set_index("Date").sort_index()
