"""Sanity checks for src/cta/backtest.py -- requires data/raw/Delta1/ to be present locally."""

import numpy as np
import pandas as pd
import pytest

from cta.backtest import (
    correlation_regime_scale,
    diversified_tsmom,
    instrument_returns,
    portfolio_mean,
)
from cta.data import curated_futures_universe
from cta.metrics import growth_of_one, sharpe_ratio

# A handful of liquid, long-history instruments across classes -- enough for a
# meaningfully non-degenerate correlation estimate without pulling in all 77.
_SAMPLE_UNIVERSE = ["&6E", "&6A", "&GC", "&SI", "&ZN", "&ZB", "&ES", "&CL"]


def test_instrument_returns_keyed_by_symbol():
    returns = instrument_returns(["&GC", "&ZN"])
    assert set(returns.keys()) == {"&GC", "&ZN"}
    assert all(isinstance(r, pd.Series) for r in returns.values())


def test_diversified_tsmom_portfolio_is_the_mean_of_its_instruments():
    result = diversified_tsmom(["&GC", "&ZN"], lookback=60, vol_window=20, cost_bps=0.0)
    per_instrument = result["per_instrument_strategy"]
    portfolio = result["strategy"]

    common_days = per_instrument.dropna().index
    for day in common_days[:5]:
        assert portfolio.loc[day] == pytest.approx(per_instrument.loc[day].mean())


# --------------------------------------------------------------------------------------
# REGRESSION: roll-gap contamination (Fix 1)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "symbol,name,max_return",
    [
        ("&VX", "VIX futures", -0.90),  # structurally decays toward -100%
        ("&CL", "WTI crude", -0.40),  # oil fell 2005-2014 and is heavily in contango
    ],
)
def test_rolling_long_position_loses_money_where_it_must(symbol, name, max_return):
    """REGRESSION for the roll-gap bug, and the single most decisive smell test.

    Under the old raw-`pct_change()` construction a rolling long VIX position "returned"
    +30.3% and long WTI +26.5% over 2005-2014 -- both impossible, and both a direct
    consequence of booking the contango roll gap as tradeable P&L.
    """
    returns = instrument_returns([symbol])[symbol].loc["2005-01-01":"2014-12-31"]
    total = growth_of_one(returns).iloc[-1] - 1
    assert total < max_return, f"{name} implied long return {total:.1%} is not physically plausible"


def test_equity_index_future_tracks_its_cash_index():
    """Validates the FIX, not just the bug: E-mini S&P has negligible roll cost, so its
    held-contract return must land near the S&P 500's ~+71% price return over 2005-2014.
    """
    returns = instrument_returns(["&ES"])["&ES"].loc["2005-01-01":"2014-12-31"]
    total = growth_of_one(returns).iloc[-1] - 1
    assert 0.55 < total < 0.90


# --------------------------------------------------------------------------------------
# REGRESSION: holiday re-levering (Fix 2)
# --------------------------------------------------------------------------------------


def test_portfolio_mean_does_not_relever_on_a_holiday():
    """REGRESSION: a market shut for a local holiday must contribute 0 P&L, not be
    dropped from the divisor. A plain skipna mean would return 0.10 here (the one open
    market standing in for the whole book) instead of the correct 0.05.
    """
    idx = pd.bdate_range("2020-01-01", periods=4)
    panel = pd.DataFrame(
        {"open_market": [0.01, 0.02, 0.10, 0.01], "holiday_market": [0.03, 0.04, np.nan, 0.05]},
        index=idx,
    )
    result = portfolio_mean(panel)
    assert result.loc[idx[2]] == pytest.approx(0.05)  # (0.10 + 0) / 2
    assert result.loc[idx[0]] == pytest.approx(0.02)  # (0.01 + 0.03) / 2


def test_portfolio_mean_excludes_instruments_before_they_exist():
    """Outside its live span an instrument is genuinely absent, so the divisor shrinks --
    this is what lets the universe ramp up over time without diluting early returns.
    """
    idx = pd.bdate_range("2020-01-01", periods=4)
    panel = pd.DataFrame(
        {"old": [0.01, 0.02, 0.03, 0.04], "new": [np.nan, np.nan, 0.10, 0.20]}, index=idx
    )
    result = portfolio_mean(panel)
    assert result.loc[idx[0]] == pytest.approx(0.01)  # only "old" is live
    assert result.loc[idx[2]] == pytest.approx(0.065)  # (0.03 + 0.10) / 2


# --------------------------------------------------------------------------------------
# Transaction costs
# --------------------------------------------------------------------------------------


def test_costs_reduce_sharpe_monotonically():
    gross = diversified_tsmom(_SAMPLE_UNIVERSE, cost_bps=0.0)["strategy"]
    cheap = diversified_tsmom(_SAMPLE_UNIVERSE, cost_bps=2.0)["strategy"]
    dear = diversified_tsmom(_SAMPLE_UNIVERSE, cost_bps=10.0)["strategy"]

    assert sharpe_ratio(gross) > sharpe_ratio(cheap) > sharpe_ratio(dear)


def test_zero_cost_matches_explicit_gross():
    a = diversified_tsmom(["&GC", "&ZN"], cost_bps=0.0)["strategy"]
    b = diversified_tsmom(["&GC", "&ZN"], cost_bps=0)["strategy"]
    pd.testing.assert_series_equal(a, b)


# --------------------------------------------------------------------------------------
# Regime layers
# --------------------------------------------------------------------------------------


def test_correlation_regime_scale_only_takes_the_two_configured_values():
    scale = correlation_regime_scale(_SAMPLE_UNIVERSE, corr_window=60, threshold_lookback=120)
    assert set(scale.dropna().unique()) <= {1.0, 0.5}


def test_correlation_regime_scale_is_nan_before_it_can_be_known():
    """REGRESSION: `NaN < threshold` is False, so the old code silently ran the book at
    half exposure through the entire warm-up instead of holding no view."""
    scale = correlation_regime_scale(_SAMPLE_UNIVERSE, corr_window=60, threshold_lookback=120)
    assert scale.iloc[:60].isna().all()


def test_correlation_regime_scale_has_no_lookahead():
    full_returns = instrument_returns(_SAMPLE_UNIVERSE)
    scale_full = correlation_regime_scale(
        _SAMPLE_UNIVERSE, corr_window=60, threshold_lookback=120, returns=full_returns
    )
    check_date = scale_full.dropna().index[200]

    cutoff = full_returns["&6E"].index[400]
    assert cutoff > check_date
    truncated = {sym: r.loc[:cutoff] for sym, r in full_returns.items()}
    scale_truncated = correlation_regime_scale(
        _SAMPLE_UNIVERSE, corr_window=60, threshold_lookback=120, returns=truncated
    )

    assert scale_full.loc[check_date] == pytest.approx(scale_truncated.loc[check_date])


def test_regime_scale_changes_the_strategy_but_not_the_benchmark():
    regime_scale = correlation_regime_scale(_SAMPLE_UNIVERSE, corr_window=60, threshold_lookback=120)
    baseline = diversified_tsmom(_SAMPLE_UNIVERSE, lookback=60, vol_window=20)
    regime_aware = diversified_tsmom(
        _SAMPLE_UNIVERSE, lookback=60, vol_window=20, regime_scale=regime_scale
    )

    pd.testing.assert_series_equal(baseline["benchmark"], regime_aware["benchmark"])
    assert not baseline["strategy"].equals(regime_aware["strategy"])


def test_trend_efficiency_changes_the_strategy_but_not_the_benchmark():
    baseline = diversified_tsmom(_SAMPLE_UNIVERSE, lookback=60, vol_window=20)
    boosted = diversified_tsmom(
        _SAMPLE_UNIVERSE,
        lookback=60,
        vol_window=20,
        trend_efficiency_window=60,
        trend_efficiency_threshold_lookback=120,
    )

    pd.testing.assert_series_equal(baseline["benchmark"], boosted["benchmark"])
    assert not baseline["strategy"].equals(boosted["strategy"])


def test_curated_universe_runs_end_to_end():
    universe = curated_futures_universe()[:10]
    result = diversified_tsmom(universe)
    assert len(result["strategy"]) > 0
    assert result["strategy"].abs().max() < 1.0  # sane daily returns, not blown up
