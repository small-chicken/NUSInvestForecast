"""Turns indicators into target positions/weights."""

import pandas as pd

from cta.indicators import momentum_signal, realized_vol


def tsmom_position(
    returns: pd.Series,
    lookback: int = 252,
    vol_window: int = 60,
    target_vol: float = 0.40,
) -> pd.Series:
    """Time-series momentum position: sign(trailing return) * target_vol / realized_vol.

    Shifted by one day so the position held on day t only uses information available
    through day t-1 (no look-ahead).
    """
    signal = momentum_signal(returns, lookback=lookback)
    vol = realized_vol(returns, window=vol_window)
    position = signal * (target_vol / vol)
    return position.shift(1)


def passive_long_position(
    returns: pd.Series,
    vol_window: int = 60,
    target_vol: float = 0.40,
) -> pd.Series:
    """Always-long benchmark position, same vol-scaling as tsmom_position -- isolates
    whether the trend *signal* adds value over just holding the same risk exposure.
    """
    vol = realized_vol(returns, window=vol_window)
    position = target_vol / vol
    return position.shift(1)
