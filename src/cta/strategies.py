"""Turns indicators into target positions/weights."""

import numpy as np
import pandas as pd

from cta.indicators import efficiency_ratio, momentum_signal, realized_vol

# Caps target_vol/vol so a realized-vol estimate that collapses near zero (e.g. a stale
# or illiquid price run) can't blow position size up to an unbounded multiple. Universe
# curation (data.curated_futures_universe) already screens out the worst offenders, but
# this is a second line of defense -- position sizing shouldn't silently explode just
# because one instrument had a quiet stretch.
#
# 20x, not 10x: an earlier 10x cap bound on ~37% of (instrument, day) observations once
# tested at full universe scale -- it wasn't just catching pathological cases, it was
# routinely throttling genuinely low-vol government bonds (e.g. 2-year notes legitimately
# trade near 2-3% annualized vol, needing ~15-20x to reach a 40% target). 20x binds on
# ~4% of observations after excluding money-market rate futures (see
# notebooks/02_strategy_research.ipynb) -- a rare backstop, not a routine constraint.
DEFAULT_MAX_LEVERAGE = 20.0


def _vol_scale(vol: pd.Series, target_vol: float, max_leverage: float) -> pd.Series:
    return (target_vol / vol).clip(upper=max_leverage)


def tsmom_position(
    returns: pd.Series,
    lookback: int = 252,
    vol_window: int = 60,
    target_vol: float = 0.40,
    max_leverage: float = DEFAULT_MAX_LEVERAGE,
) -> pd.Series:
    """Time-series momentum position: sign(trailing return) * target_vol / realized_vol.

    Shifted by one day so the position held on day t only uses information available
    through day t-1 (no look-ahead).
    """
    signal = momentum_signal(returns, lookback=lookback)
    vol = realized_vol(returns, window=vol_window)
    position = signal * _vol_scale(vol, target_vol, max_leverage)
    return position.shift(1)


def passive_long_position(
    returns: pd.Series,
    vol_window: int = 60,
    target_vol: float = 0.40,
    max_leverage: float = DEFAULT_MAX_LEVERAGE,
) -> pd.Series:
    """Always-long benchmark position, same vol-scaling as tsmom_position -- isolates
    whether the trend *signal* adds value over just holding the same risk exposure.
    """
    vol = realized_vol(returns, window=vol_window)
    position = _vol_scale(vol, target_vol, max_leverage)
    return position.shift(1)


def trend_efficiency_scale(
    returns: pd.Series,
    window: int = 252,
    threshold_lookback: int = 504,
    baseline: float = 1.0,
    boost: float = 1.5,
) -> pd.Series:
    """Position multiplier that boosts exposure specifically during unusually smooth,
    persistent trends (high `efficiency_ratio`) -- targeting the exact mechanism found in
    notebooks/02_strategy_research.ipynb: vol-scaled position sizing shrinks whenever
    realized vol rises, even when that's happening *within* an intact, correctly-called
    trend rather than ahead of a reversal, so a real uptrend's fastest legs get
    under-participated in. Boosting specifically when the trend has been efficient (not
    just trending) is a targeted response to that finding, not a general-purpose regime
    signal borrowed from elsewhere (contrast the correlation overlay built on
    `indicators.average_pairwise_correlation`).

    `baseline` (default 1.0), not a cut below baseline: this is meant to claw back
    upside the vol-scaling gave up, not to add a second risk-reduction dial on top of
    `strategies.tsmom_position`'s existing vol-scaling and leverage cap.

    Self-calibrating (trailing median, not a fixed ER threshold) and shifted by one day,
    same discipline as every other position function here.
    """
    er = efficiency_ratio(returns, window=window)
    threshold = er.rolling(threshold_lookback, min_periods=window).median()
    scale = pd.Series(np.where(er > threshold, boost, baseline), index=er.index)
    return scale.shift(1)
