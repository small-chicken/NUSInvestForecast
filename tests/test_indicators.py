"""Sanity checks for src/cta/indicators.py and strategies.py."""

import numpy as np
import pandas as pd
import pytest

from cta.indicators import daily_returns, momentum_signal, realized_vol
from cta.strategies import passive_long_position, tsmom_position


def _price_series(n, daily_return, start=100.0):
    idx = pd.bdate_range("2020-01-01", periods=n)
    returns = pd.Series(daily_return, index=idx)
    return start * (1 + returns).cumprod()


def test_momentum_signal_flips_on_trend_reversal():
    up = _price_series(n=300, daily_return=0.002)
    down = _price_series(n=300, daily_return=-0.002, start=up.iloc[-1])
    down.index = pd.bdate_range(up.index[-1] + pd.Timedelta(days=1), periods=len(down))

    returns = daily_returns(pd.concat([up, down]))
    signal = momentum_signal(returns, lookback=252).dropna()

    assert signal.iloc[0] == 1.0  # still in the up-trend, 252 days after it started
    assert signal.iloc[-1] == -1.0  # 252 days into the down-trend


def test_realized_vol_matches_known_value():
    idx = pd.bdate_range("2020-01-01", periods=200)
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    vol = realized_vol(returns, window=60)
    expected = returns.iloc[-60:].std() * np.sqrt(252)
    assert vol.iloc[-1] == pytest.approx(expected)


def test_tsmom_position_has_no_lookahead():
    prices = _price_series(n=400, daily_return=0.001)
    returns = daily_returns(prices)
    position_full = tsmom_position(returns, lookback=100, vol_window=30)

    # Truncate the return series partway through and recompute. If a position on a date
    # well before the cutoff changes because of this, future data leaked backward.
    cutoff = returns.index[300]
    position_truncated = tsmom_position(returns.loc[:cutoff], lookback=100, vol_window=30)

    check_date = returns.index[250]
    assert position_full.loc[check_date] == pytest.approx(position_truncated.loc[check_date])


def test_passive_position_always_same_sign():
    prices = _price_series(n=300, daily_return=-0.001)
    returns = daily_returns(prices)
    position = passive_long_position(returns, vol_window=30).dropna()
    assert (position > 0).all()
