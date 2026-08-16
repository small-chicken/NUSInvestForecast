"""Sanity checks for src/cta/indicators.py and strategies.py."""

import numpy as np
import pandas as pd
import pytest

from cta.indicators import (
    average_pairwise_correlation,
    daily_returns,
    efficiency_ratio,
    held_contract_returns,
    momentum_signal,
    naive_spliced_returns,
    realized_vol,
)
from cta.strategies import (
    blended_momentum_signal,
    passive_long_position,
    trend_efficiency_scale,
    tsmom_position,
)


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


def test_position_is_capped_when_vol_collapses_near_zero():
    # Reproduces the &YIB bug: a long stale (unchanged) price run collapses realized_vol
    # toward zero, which without a cap sends target_vol/vol -- and the resulting
    # "return" -- into the billions the moment the price moves again.
    idx = pd.bdate_range("2020-01-01", periods=200)
    prices = pd.Series(100.0, index=idx)
    prices.iloc[100:] = 100.0  # already flat; explicit for clarity
    prices.iloc[150:] = 100.5  # one real move after the stale stretch
    returns = daily_returns(prices)

    position = tsmom_position(returns, lookback=60, vol_window=60, max_leverage=10.0)
    assert position.dropna().abs().max() <= 10.0


def _correlated_panel(n_instruments, n_days, rho, seed, scales=None):
    """Common-factor model: pairwise corr(x_i, x_j) = rho for every i != j."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    common = rng.normal(0, 1, n_days)
    cols = {}
    for i in range(n_instruments):
        series = np.sqrt(rho) * common + np.sqrt(1 - rho) * rng.normal(0, 1, n_days)
        # scale each column differently -- correlation must be invariant to this
        cols[f"inst_{i}"] = series * (scales[i] if scales is not None else 1.0)
    return pd.DataFrame(cols, index=idx)


def test_average_pairwise_correlation_recovers_known_correlation():
    panel = _correlated_panel(20, 2000, rho=0.35, seed=0)
    corr = average_pairwise_correlation(panel, window=500, step=21).dropna()
    assert corr.iloc[-1] == pytest.approx(0.35, abs=0.05)


def test_average_pairwise_correlation_near_zero_for_independent_series():
    panel = _correlated_panel(15, 1000, rho=0.0, seed=1)
    corr = average_pairwise_correlation(panel, window=500, step=21).dropna()
    assert corr.iloc[-1] == pytest.approx(0.0, abs=0.05)


def test_average_pairwise_correlation_survives_nonunit_variance_and_nans():
    """REGRESSION: the two conditions the previous identity-based estimator broke on.

    The old implementation used Var(mean) = 1/N + (N-1)/N*rho, which is only valid for
    unit-variance columns, and used the full column count regardless of how many were
    actually observed. Its production caller fed it annualized-vol-normalized data (std
    ~ 1/sqrt(252)) with ragged NaNs, so it silently returned a constant -1/(N-1) for the
    entire sample. The old tests passed only because they fed unit-variance NaN-free data.
    """
    rng = np.random.default_rng(7)
    n_instruments, rho = 12, 0.40
    scales = rng.uniform(0.01, 50.0, n_instruments)  # wildly non-unit, wildly unequal
    panel = _correlated_panel(n_instruments, 1200, rho=rho, seed=3, scales=scales)
    # ragged histories: each instrument starts at a different date, plus holiday gaps
    for i, col in enumerate(panel.columns):
        panel.iloc[: i * 20, panel.columns.get_loc(col)] = np.nan
    panel.iloc[500:505, 0] = np.nan

    corr = average_pairwise_correlation(panel, window=400, step=21).dropna()
    assert corr.iloc[-1] == pytest.approx(rho, abs=0.06)
    # and it must not be the degenerate constant the old estimator produced
    assert corr.nunique() > 5


def test_average_pairwise_correlation_matches_naive_reference():
    """Match a direct O(N^2) reference implementation on the same window."""
    panel = _correlated_panel(10, 600, rho=0.25, seed=11, scales=np.linspace(0.1, 10, 10))
    window = 300
    corr = average_pairwise_correlation(panel, window=window, step=1, min_overlap=30)

    block = panel.iloc[-window:]
    reference_matrix = block.corr().to_numpy()
    off_diagonal = ~np.eye(len(reference_matrix), dtype=bool)
    expected = reference_matrix[off_diagonal].mean()

    assert corr.iloc[-1] == pytest.approx(expected, abs=1e-9)


def test_efficiency_ratio_is_high_for_a_monotonic_trend():
    prices = _price_series(n=300, daily_return=0.001)  # every day up -> no reversal at all
    returns = daily_returns(prices)
    er = efficiency_ratio(returns, window=200).dropna()
    assert er.iloc[-1] == pytest.approx(1.0, abs=1e-6)


def test_efficiency_ratio_is_low_for_a_choppy_flat_series():
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(2)
    # zero-drift noise: net move should be small relative to the total distance travelled
    prices = pd.Series(100 * (1 + rng.normal(0, 0.01, 300)).cumprod(), index=idx)
    returns = daily_returns(prices)
    er = efficiency_ratio(returns, window=200).dropna()
    assert er.iloc[-1] < 0.3


def test_trend_efficiency_scale_boosts_once_a_choppy_period_turns_smooth():
    idx_choppy = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(3)
    choppy_prices = pd.Series(100 * (1 + rng.normal(0, 0.01, 400)).cumprod(), index=idx_choppy)
    smooth_prices = _price_series(n=400, daily_return=0.002, start=choppy_prices.iloc[-1])
    smooth_prices.index = pd.bdate_range(idx_choppy[-1] + pd.Timedelta(days=1), periods=400)

    returns = daily_returns(pd.concat([choppy_prices, smooth_prices]))
    scale = trend_efficiency_scale(returns, window=100, threshold_lookback=300).dropna()
    assert scale.isin([1.0, 1.5]).all()

    # Shortly after the transition (window has rolled past the choppy->smooth switch,
    # but the 300-day trailing median still spans mostly-choppy history), ER should
    # clearly exceed its own trailing median and the boost should engage. (Far past the
    # transition the median itself catches up to the now-constant ER and the boost drops
    # back out -- an expected edge case, not what this test is checking.)
    check_date = returns.index[400 + 150]
    assert scale.loc[check_date] == 1.5


def test_trend_efficiency_scale_has_no_lookahead():
    prices = _price_series(n=800, daily_return=0.0008)
    returns = daily_returns(prices)
    scale_full = trend_efficiency_scale(returns, window=100, threshold_lookback=300)
    cutoff = returns.index[600]
    scale_truncated = trend_efficiency_scale(returns.loc[:cutoff], window=100, threshold_lookback=300)

    check_date = returns.index[500]
    assert scale_full.loc[check_date] == pytest.approx(scale_truncated.loc[check_date])


def test_held_contract_returns_excludes_the_roll_gap():
    """The back-adjusted series differs from raw by a per-roll constant, so on a
    non-roll day both give the same return, and on a roll day only the naive one
    books the (untradeable) calendar spread."""
    idx = pd.bdate_range("2020-01-01", periods=5)
    raw = pd.Series([100.0, 101.0, 90.0, 91.0, 92.0], index=idx)  # -11 jump = the roll
    ccb = pd.Series([110.0, 111.0, 111.0, 112.0, 113.0], index=idx)  # roll spliced out

    held = held_contract_returns(ccb, raw)
    naive = naive_spliced_returns(raw)

    assert held.loc[idx[1]] == pytest.approx(1.0 / 100.0)  # same as naive off-roll
    assert naive.loc[idx[1]] == pytest.approx(1.0 / 100.0)
    assert held.loc[idx[2]] == pytest.approx(0.0)  # roll day: no real move
    assert naive.loc[idx[2]] == pytest.approx(-11.0 / 101.0)  # ...booked as -10.9%


# --------------------------------------------------------------------------------------
# Multi-horizon momentum blend (tested, not adopted -- see notebooks/02, Part 4)
# --------------------------------------------------------------------------------------


def test_blended_signal_is_full_size_when_every_horizon_agrees():
    steady_uptrend = pd.Series(
        0.001, index=pd.bdate_range("2015-01-01", periods=400)
    )
    blended = blended_momentum_signal(steady_uptrend, [21, 63, 252])
    assert blended.dropna().iloc[-1] == pytest.approx(1.0)


def test_blended_signal_scales_down_when_horizons_disagree():
    """A short rally inside a long downtrend: the 1-month signal turns positive while the
    12-month one stays negative, so the blend must land strictly between -1 and +1."""
    idx = pd.bdate_range("2015-01-01", periods=400)
    path = np.concatenate([np.full(390, -0.003), np.full(10, 0.02)])
    returns = pd.Series(path, index=idx)

    blended = blended_momentum_signal(returns, [21, 252]).dropna()
    assert -1.0 < blended.iloc[-1] < 1.0


def test_tsmom_position_accepts_a_single_horizon_or_a_sequence():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=500)
    returns = pd.Series(rng.normal(0.0004, 0.01, 500), index=idx)

    single = tsmom_position(returns, lookback=252, vol_window=60)
    blended = tsmom_position(returns, lookback=[21, 63, 252], vol_window=60)

    assert not single.equals(blended)
    # the blend can never demand a bigger position than full conviction at the same vol
    assert blended.abs().max() <= single.abs().max() + 1e-12
