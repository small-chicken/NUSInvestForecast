"""Sanity checks for src/cta/metrics.py."""

import numpy as np
import pandas as pd
import pytest

from cta.metrics import (
    apply_costs,
    cumulative_return,
    growth_of_one,
    max_drawdown,
    sharpe_ratio,
    turnover,
)


def test_sharpe_ratio_of_known_series():
    idx = pd.bdate_range("2020-01-01", periods=252)
    rng = np.random.default_rng(0)
    returns = pd.Series(0.0005 + rng.normal(0, 0.001, len(idx)), index=idx)
    expected = returns.mean() / returns.std() * np.sqrt(252)
    assert sharpe_ratio(returns) == pytest.approx(expected)


def test_growth_of_one_and_cumulative_return_agree():
    returns = pd.Series([0.10, -0.10, 0.05])
    growth = growth_of_one(returns)
    assert growth.iloc[-1] == pytest.approx(1.10 * 0.90 * 1.05)
    assert cumulative_return(returns) == pytest.approx(growth.iloc[-1] - 1)


def test_max_drawdown_matches_hand_computed_value():
    # wealth path: 1 -> 1.20 -> 0.90 -> 1.00  (peak 1.20, trough 0.90 -> -25% drawdown)
    returns = pd.Series([0.20, -0.25, 0.1111])
    assert max_drawdown(returns) == pytest.approx(-0.25, abs=1e-3)


def test_max_drawdown_is_zero_for_monotonic_gains():
    returns = pd.Series([0.01, 0.02, 0.01, 0.03])
    assert max_drawdown(returns) == pytest.approx(0.0)


def test_max_drawdown_is_floored_at_minus_one():
    """REGRESSION: a levered wealth path can cross zero, after which the naive
    wealth/cummax-1 formula produces meaningless values (-284% was observed on real
    single-instrument legs). Once wealth hits zero the account is gone: -100%."""
    wiped_out = pd.Series([0.5, -1.5, 0.3])  # wealth goes 1.5 -> -0.75
    assert max_drawdown(wiped_out) == pytest.approx(-1.0)


def test_turnover_counts_absolute_position_changes():
    positions = pd.DataFrame({"a": [1.0, 2.0, 2.0], "b": [0.0, -1.0, -1.0]})
    t = turnover(positions)
    assert t.iloc[1] == pytest.approx(2.0)  # |2-1| + |-1-0|
    assert t.iloc[2] == pytest.approx(0.0)  # no trading


def test_apply_costs_charges_only_when_trading():
    returns = pd.Series([0.01, 0.01, 0.01])
    positions = pd.DataFrame({"a": [1.0, 2.0, 2.0]})
    net = apply_costs(returns, positions, cost_bps=100.0)  # 1% per unit traded
    assert net.iloc[1] == pytest.approx(0.01 - 0.01)  # traded 1 unit
    assert net.iloc[2] == pytest.approx(0.01)  # no trade, no cost
