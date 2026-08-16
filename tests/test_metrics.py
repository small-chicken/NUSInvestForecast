"""Sanity checks for src/cta/metrics.py."""

import numpy as np
import pandas as pd
import pytest

from cta.metrics import (
    apply_costs,
    block_bootstrap,
    cumulative_return,
    growth_of_one,
    max_drawdown,
    rolling_sharpe,
    sharpe_difference_interval,
    sharpe_interval,
    sharpe_ratio,
    t_statistic,
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


# --------------------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------------------


def _noise(n=1500, mean=0.0, seed=0):
    idx = pd.bdate_range("2005-01-01", periods=n)
    rng = np.random.default_rng(seed)
    return pd.Series(mean + rng.normal(0, 0.01, n), index=idx)


def test_t_statistic_is_sharpe_times_root_years():
    returns = _noise(mean=0.0004)
    years = len(returns) / 252
    assert t_statistic(returns) == pytest.approx(sharpe_ratio(returns) * np.sqrt(years))


def test_block_bootstrap_returns_one_value_per_resample():
    draws = block_bootstrap(_noise().to_frame("r"), lambda d: sharpe_ratio(d["r"]), n_resamples=50)
    assert draws.shape == (50,)


def test_block_bootstrap_is_reproducible_given_a_seed():
    frame = _noise().to_frame("r")

    def stat(d):
        return sharpe_ratio(d["r"])

    a = block_bootstrap(frame, stat, n_resamples=40, seed=7)
    b = block_bootstrap(frame, stat, n_resamples=40, seed=7)
    np.testing.assert_allclose(a, b)


def test_block_bootstrap_resamples_are_the_length_of_the_original():
    """Blocks are wrapped, not truncated, so every resample is a full-length sample --
    otherwise shorter resamples would inflate the apparent spread of the statistic."""
    frame = _noise(n=1000).to_frame("r")
    lengths = block_bootstrap(frame, len, n_resamples=25)
    assert set(lengths) == {1000}


def test_block_bootstrap_uses_the_whole_sample_including_its_ends():
    """REGRESSION for the circular part of "circular block bootstrap". A plain moving-block
    bootstrap can never place the final observations at the start of a block, so extreme
    values living at the edges of the sample (2008, here) are under-sampled."""
    frame = pd.DataFrame({"r": [0.0] * 99 + [1.0]})  # the only non-zero value is last
    sums = block_bootstrap(frame, lambda d: float(d["r"].sum()), n_resamples=200, block=5)
    assert (sums > 0).mean() > 0.5


def test_sharpe_interval_brackets_its_own_point_estimate():
    returns = _noise(mean=0.0006)
    result = sharpe_interval(returns, n_resamples=400)
    assert result["ci_low"] < result["estimate"] < result["ci_high"]
    assert result["t_stat"] == pytest.approx(t_statistic(returns))


def test_sharpe_interval_of_a_zero_mean_series_straddles_zero():
    """Demeaned, so the SAMPLE Sharpe is exactly zero rather than merely drawn from a
    zero-mean population -- a random draw can easily have a visibly positive mean, and
    asserting on the population would make this test flaky rather than meaningful."""
    noise = _noise(seed=3)
    zero_mean = noise - noise.mean()

    result = sharpe_interval(zero_mean, n_resamples=400)
    assert result["estimate"] == pytest.approx(0.0)
    assert result["ci_low"] < 0 < result["ci_high"]
    assert 0.3 < result["p_positive"] < 0.7


def test_sharpe_difference_interval_is_centred_on_the_actual_difference():
    a, b = _noise(mean=0.0008, seed=1), _noise(mean=0.0002, seed=2)
    result = sharpe_difference_interval(a, b, n_resamples=400)
    assert result["estimate"] == pytest.approx(sharpe_ratio(a) - sharpe_ratio(b))
    assert result["ci_low"] < result["estimate"] < result["ci_high"]


def test_sharpe_difference_interval_keeps_the_two_legs_paired():
    """Two identical series must have a difference of exactly zero in EVERY resample.
    That only holds if both legs are resampled on the same days -- if the bootstrap drew
    them independently the difference would wander, and every "A beats B" interval in the
    write-up would be too wide."""
    series = _noise(seed=5)
    result = sharpe_difference_interval(series, series, n_resamples=200)
    assert result["estimate"] == pytest.approx(0.0)
    assert result["ci_low"] == pytest.approx(0.0)
    assert result["ci_high"] == pytest.approx(0.0)


def test_block_bootstrap_rejects_a_sample_shorter_than_one_block():
    with pytest.raises(ValueError, match="at least"):
        block_bootstrap(pd.DataFrame({"r": [0.01] * 5}), lambda d: d["r"].mean(), block=21)


def test_rolling_sharpe_matches_a_hand_computed_window():
    returns = _noise(n=400)
    rolling = rolling_sharpe(returns, window=100)
    window = returns.iloc[:100]
    expected = window.mean() / window.std() * np.sqrt(252)
    assert rolling.iloc[0] == pytest.approx(expected)
