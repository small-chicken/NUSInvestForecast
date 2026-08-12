"""Signal building blocks: moving averages, vol estimators, z-scores, regime classifiers."""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def daily_returns(close: pd.Series) -> pd.Series:
    """Simple daily % returns from a raw (non-back-adjusted) continuous contract close."""
    return close.pct_change().dropna()


def realized_vol(returns: pd.Series, window: int = 60) -> pd.Series:
    """Annualized realized volatility: rolling std of daily returns, scaled by sqrt(252).

    Simpler than MOP's exponentially-weighted estimator (see research/literature_notes.md)
    -- a documented, optional refinement, not required for a baseline.
    """
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def momentum_signal(returns: pd.Series, lookback: int = 252) -> pd.Series:
    """Sign of the trailing `lookback`-day compounded return (Moskowitz-Ooi-Pedersen TSMOM).

    +1 if the instrument is up over the lookback window, -1 if down, 0 if exactly flat.
    """
    cum_return = (1 + returns).cumprod()
    trailing_return = cum_return / cum_return.shift(lookback) - 1
    return np.sign(trailing_return)
