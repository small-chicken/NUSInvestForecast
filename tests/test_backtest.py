"""Sanity checks for src/cta/backtest.py -- requires data/raw/Delta1/ to be present locally."""

import numpy as np
import pandas as pd
import pytest

from cta.backtest import (
    EQUITY_SYMBOL,
    blend,
    correlation_ranked_universe,
    correlation_regime_scale,
    diversified_tsmom,
    instrument_returns,
    portfolio_leverage,
    portfolio_mean,
    reference_books,
    universe_breadth_study,
)
from cta.data import curated_futures_universe
from cta.metrics import annualized_volatility, growth_of_one, sharpe_ratio

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


# --------------------------------------------------------------------------------------
# Book-level volatility targeting
# --------------------------------------------------------------------------------------


def test_portfolio_leverage_de_levers_when_volatility_rises():
    calm = pd.Series(0.001, index=pd.bdate_range("2020-01-01", periods=300))
    rng = np.random.default_rng(0)
    noisy = calm + pd.Series(rng.normal(0, 0.02, 300), index=calm.index)

    steady = portfolio_leverage(calm + 1e-4 * rng.normal(0, 1, 300), target_vol=0.10, window=60)
    turbulent = portfolio_leverage(noisy, target_vol=0.10, window=60)

    assert turbulent.dropna().mean() < steady.dropna().mean()


def test_portfolio_leverage_is_capped():
    almost_flat = pd.Series(
        1e-9, index=pd.bdate_range("2020-01-01", periods=300)
    ) * np.arange(300)
    leverage = portfolio_leverage(almost_flat, target_vol=0.10, window=60, max_leverage=3.0)
    assert leverage.dropna().max() <= 3.0


def test_portfolio_leverage_has_no_lookahead():
    """Leverage applied on day t must be computable from returns through t-1, so truncating
    the series after t cannot change it."""
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=500)
    returns = pd.Series(rng.normal(0.0003, 0.01, 500), index=idx)

    full = portfolio_leverage(returns, target_vol=0.10, window=100)
    check_date = idx[300]
    truncated = portfolio_leverage(returns.loc[:check_date], target_vol=0.10, window=100)

    assert full.loc[check_date] == pytest.approx(truncated.loc[check_date])


def test_portfolio_vol_target_moves_realized_vol_toward_the_target():
    plain = diversified_tsmom(_SAMPLE_UNIVERSE, cost_bps=0.0)["strategy"]
    targeted = diversified_tsmom(_SAMPLE_UNIVERSE, cost_bps=0.0, portfolio_vol_target=0.10)[
        "strategy"
    ]
    common = plain.index.intersection(targeted.index)

    assert abs(annualized_volatility(targeted.loc[common]) - 0.10) < abs(
        annualized_volatility(plain.loc[common]) - 0.10
    )


def test_portfolio_vol_target_charges_for_the_relevering_trades():
    """REGRESSION: leverage is applied to POSITIONS, not to the finished return series.
    Scaling returns would capture the P&L of de-levering into a crisis while treating the
    trades that achieve it as free -- so the overlay would look better than it is."""
    gross = diversified_tsmom(_SAMPLE_UNIVERSE, cost_bps=0.0, portfolio_vol_target=0.10)
    net = diversified_tsmom(_SAMPLE_UNIVERSE, cost_bps=5.0, portfolio_vol_target=0.10)

    positions = gross["strategy_positions"]
    leverage = gross["strategy_leverage"]
    # positions really are levered, and the extra turnover really is charged
    assert leverage.dropna().std() > 0
    assert positions.abs().sum(axis=1).corr(leverage.reindex(positions.index)) > 0.1
    assert net["strategy"].mean() < gross["strategy"].mean()


def test_portfolio_vol_target_applies_to_both_legs():
    """The benchmark exists to hold risk constant while the signal varies. An overlay that
    reshaped the risk of one leg only would break exactly that comparison."""
    plain = diversified_tsmom(_SAMPLE_UNIVERSE, cost_bps=0.0)
    targeted = diversified_tsmom(_SAMPLE_UNIVERSE, cost_bps=0.0, portfolio_vol_target=0.10)

    assert not plain["benchmark"].equals(targeted["benchmark"])
    assert targeted["benchmark_leverage"] is not None
    assert plain["strategy_leverage"] is None


# --------------------------------------------------------------------------------------
# REGRESSION: the "live instrument" span must not look forward (Fix 3)
# --------------------------------------------------------------------------------------


def test_portfolio_mean_does_not_use_future_data_to_decide_who_is_live():
    """REGRESSION: the live span used to end at an instrument's LAST observation, found
    with a reverse cumulative max -- i.e. "is there any print at or after today?", which
    a backtester standing on that day cannot know. Truncating the panel must not change
    any weight computed before the truncation point."""
    idx = pd.bdate_range("2020-01-01", periods=60)
    panel = pd.DataFrame(
        {"survivor": 0.01, "delisted": [0.02] * 30 + [np.nan] * 30}, index=idx
    )

    full = portfolio_mean(panel)
    truncated = portfolio_mean(panel.iloc[:20])
    pd.testing.assert_series_equal(full.iloc[:20], truncated, check_freq=False)


def test_portfolio_mean_drops_an_instrument_after_it_goes_quiet():
    idx = pd.bdate_range("2020-01-01", periods=60)
    panel = pd.DataFrame({"live": 0.01, "delisted": [0.03] * 10 + [np.nan] * 50}, index=idx)

    result = portfolio_mean(panel, delist_after=5)
    assert result.iloc[0] == pytest.approx(0.02)  # both live: (0.01 + 0.03) / 2
    assert result.iloc[12] == pytest.approx(0.005)  # still counted, contributing 0
    assert result.iloc[40] == pytest.approx(0.01)  # long gone, out of the divisor


# --------------------------------------------------------------------------------------
# Reference books and universe studies
# --------------------------------------------------------------------------------------


def test_reference_books_are_plain_unlevered_holdings():
    books = reference_books()
    equity = instrument_returns([EQUITY_SYMBOL])[EQUITY_SYMBOL]

    pd.testing.assert_series_equal(books["Equity (S&P futures)"], equity, check_names=False)
    # 60/40 must be less volatile than 100% equity, or it isn't a 60/40
    common = books["60/40 equity/bonds"].index.intersection(equity.index)
    assert books["60/40 equity/bonds"].loc[common].std() < equity.loc[common].std()


def test_blend_interpolates_between_its_two_legs():
    idx = pd.bdate_range("2020-01-01", periods=10)
    core = pd.Series(0.01, index=idx)
    sleeve = pd.Series(0.03, index=idx)

    assert blend(core, sleeve, 0.0).iloc[0] == pytest.approx(0.01)
    assert blend(core, sleeve, 1.0).iloc[0] == pytest.approx(0.03)
    assert blend(core, sleeve, 0.25).iloc[0] == pytest.approx(0.015)


def test_universe_breadth_study_reports_one_row_per_draw():
    study = universe_breadth_study(
        _SAMPLE_UNIVERSE, sizes=[2, 4], window=slice("2005-01-01", "2014-12-31"), n_draws=3
    )
    assert len(study) == 6
    assert set(study["n_markets"]) == {2, 4}


def test_universe_breadth_study_evaluates_the_full_universe_once():
    """A "random draw" from the whole universe is just the universe, so drawing it
    repeatedly would plot the same point N times and imply a spread that isn't there."""
    study = universe_breadth_study(
        _SAMPLE_UNIVERSE,
        sizes=[len(_SAMPLE_UNIVERSE)],
        window=slice("2005-01-01", "2014-12-31"),
        n_draws=5,
    )
    assert len(study) == 1


def test_correlation_ranked_universe_uses_only_data_before_the_cutoff():
    """The whole point of ranking point-in-time: a selection made at end-2009 must not
    move when 2010-2014 data is appended."""
    returns = instrument_returns(_SAMPLE_UNIVERSE)
    truncated = {s: r.loc[:"2009-12-31"] for s, r in returns.items()}

    from_full = correlation_ranked_universe(_SAMPLE_UNIVERSE, 4, "2009-12-31", returns=returns)
    from_truncated = correlation_ranked_universe(
        _SAMPLE_UNIVERSE, 4, "2009-12-31", returns=truncated
    )
    assert from_full == from_truncated


def test_correlation_ranked_universe_ends_pick_opposite_markets():
    most = correlation_ranked_universe(_SAMPLE_UNIVERSE, 3, "2009-12-31", most_correlated=True)
    least = correlation_ranked_universe(_SAMPLE_UNIVERSE, 3, "2009-12-31", most_correlated=False)
    assert not set(most) & set(least)
