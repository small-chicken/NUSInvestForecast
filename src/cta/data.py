"""Loading, cleaning, and aligning raw price data from data/raw/Delta1/."""

from __future__ import annotations

from functools import lru_cache
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


@lru_cache(maxsize=256)
def load_futures(symbol: str) -> pd.DataFrame:
    """OHLCV + open interest for one futures symbol, indexed by Date.

    `symbol` should include the leading `&` (raw continuous contract) or
    `&..._CCB` (back-adjusted continuous contract), matching the catalogue.

    Cached: the walk-forward re-derives the universe at every year boundary, which would
    otherwise re-read every CSV a few thousand times. Callers must not mutate the frame
    they get back.
    """
    df = pd.read_csv(FUTURES_DIR / f"{symbol}.csv", parse_dates=["Date"])
    return df.set_index("Date").sort_index()


def load_etf(symbol: str) -> pd.DataFrame:
    """OHLCV + dividends for one ETF symbol, indexed by Date."""
    df = pd.read_csv(ETF_DIR / f"{symbol}.csv", parse_dates=["Date"])
    return df.set_index("Date").sort_index()


# Excluded from the curated universe (see notebooks/01_data_exploration.ipynb):
# - starts after 2004-12-31, too little history before a 2010-2014 OOS window
_LATE_STARTERS = {"&6N", "&EUA", "&FDAX9", "&FESX9", "&FGBX", "&FTDX", "&RB", "&WBS"}
# - redundant contract variants of an already-included instrument
_REDUNDANT = {"&YAP4", "&YAP10"}

# A run of unchanged closing prices this long or longer marks a market as too illiquid
# to trust for vol-scaled position sizing -- discovered when &YIB (163 stale days) sent
# realized_vol to ~1e-10 and blew position sizing up to billions (see
# notebooks/02_strategy_research.ipynb). MOP's own paper explicitly restricts to liquid
# instruments for the same reason.
_STALE_RUN_THRESHOLD_DAYS = 20

# Short-term money-market rate futures (interbank/bank-bill rates), not government bond
# duration futures. MOP's own universe only ever includes "-year" duration bonds (2/5/10/
# 30yr Treasuries, Bunds, Gilts, ...) -- these trade near-frozen for weeks at a time by
# their nature (a 90-day rate expectation barely moves day to day), which is economically
# real, not a data error, but structurally mismatched with a uniform vol-target framework
# built for directional trend markets. Excluded on the same grounds MOP's universe does
# implicitly (see notebooks/02_strategy_research.ipynb).
_MONEY_MARKET_RATES = {"&BAX", "&LEU", "&YIR"}

# Minimum trading days of history required before a market can be traded, when selecting
# point-in-time: 252 for the momentum lookback + 60 for the vol estimate, plus headroom.
_MIN_HISTORY_DAYS = 400


def _longest_stale_run(close: pd.Series) -> int:
    """Longest run of consecutive unchanged (stale) closing prices."""
    unchanged = close.diff() == 0
    if not unchanged.any():
        return 0
    run_id = (~unchanged).cumsum()
    return int(unchanged.groupby(run_id).sum().max())


def curated_futures_universe(as_of: str | pd.Timestamp | None = None) -> list[str]:
    """Diversified futures root symbols (raw, non-`_CCB`) used for strategy work.

    Excludes redundant contract variants (e.g. multiple ASX SPI 200 listings), markets
    with long stale-price runs that make vol-scaled position sizing unsafe, short-term
    money-market rate futures (a structurally different instrument class from the
    government bond futures a diversified trend book targets), and markets with too
    little history to trade -- see notebooks/01_data_exploration.ipynb and
    notebooks/02_strategy_research.ipynb for the analysis behind this list.

    `as_of` makes the screen POINT-IN-TIME: only price history up to that date is used to
    judge staleness and sufficient history, so the universe is one a backtester standing
    at `as_of` could actually have selected. Leaving it None reproduces the original
    full-sample screen, which is convenient for comparison but is a mild look-ahead --
    the selection rules were themselves chosen after seeing the whole sample.
    """
    catalogue = load_futures_catalogue()
    roots = catalogue.loc[~catalogue["symbol"].str.endswith("_CCB"), "symbol"]
    cutoff = pd.Timestamp(as_of) if as_of is not None else None

    excluded = set(_REDUNDANT) | set(_MONEY_MARKET_RATES)
    for sym in roots:
        close = load_futures(sym)["Close"]
        if cutoff is not None:
            close = close.loc[:cutoff]
        # Needs enough history before `as_of` to form a 12-month signal plus vol estimate.
        if cutoff is not None and len(close) < _MIN_HISTORY_DAYS:
            excluded.add(sym)
        if _longest_stale_run(close) >= _STALE_RUN_THRESHOLD_DAYS:
            excluded.add(sym)
    if cutoff is None:
        excluded |= set(_LATE_STARTERS)

    return sorted(set(roots) - excluded)
