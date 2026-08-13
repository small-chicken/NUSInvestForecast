"""Combine per-instrument positions into a diversified portfolio backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cta import data
from cta.indicators import (
    average_pairwise_correlation,
    held_contract_returns,
    naive_spliced_returns,
)
from cta.metrics import apply_costs
from cta.strategies import passive_long_position, trend_efficiency_scale, tsmom_position

DEFAULT_COST_BPS = 2.0


def instrument_returns(
    symbols: list[str], construction: str = "held_contract"
) -> dict[str, pd.Series]:
    """Daily returns per futures symbol, each on its own native calendar.

    Computed per-instrument before any cross-instrument combination -- combining first
    would let one instrument's holiday poison a rolling-window calculation for another
    (see notebooks/01_data_exploration.ipynb).

    `construction`:
    - "held_contract" (default, correct): return of holding a single contract, roll gaps
      excluded. See `indicators.held_contract_returns`.
    - "naive_spliced" (INCORRECT): percentage change straight off the raw spliced
      continuous contract, which books roll gaps as tradeable P&L. Retained only so
      notebooks/02_strategy_research.ipynb can reproduce the original contaminated result
      side by side with the fix.
    """
    if construction not in {"held_contract", "naive_spliced"}:
        raise ValueError(f"unknown construction {construction!r}")

    out = {}
    for sym in symbols:
        raw = data.load_futures(sym)["Close"]
        if construction == "naive_spliced":
            out[sym] = naive_spliced_returns(raw)
        else:
            out[sym] = held_contract_returns(data.load_futures(sym + "_CCB")["Close"], raw)
    return out


def portfolio_mean(per_instrument: pd.DataFrame) -> pd.Series:
    """Equal-weight across LIVE instruments, booking closed markets at 0 P&L.

    A plain `.mean(axis=1)` uses pandas' default skipna=True, which drops a market that is
    shut for a local holiday from the DIVISOR rather than crediting it 0 P&L -- so every
    open market's weight jumps from 1/N_live to 1/N_open. In 2005-2014 that silently
    re-levered the book on 540 partial days, and on 7 of them booked a "77-market
    diversified portfolio" as 3 markets at a 40% vol target each.

    An instrument is "live" from its first to its last observation. Inside that span a
    missing value means the exchange was closed, so the position simply earns nothing.
    Outside it the instrument does not exist yet (or has expired) and is correctly absent
    from the denominator -- which is what lets the universe ramp up over time.
    """
    valid = per_instrument.notna()
    started = valid.cummax()
    ended = valid[::-1].cummax()[::-1]
    live = started & ended

    n_live = live.sum(axis=1)
    contributions = per_instrument.where(valid, 0.0).where(live, 0.0)
    return (contributions.sum(axis=1) / n_live.replace(0, np.nan)).dropna()


def diversified_tsmom(
    symbols: list[str],
    lookback: int = 252,
    vol_window: int = 60,
    target_vol: float = 0.40,
    regime_scale: pd.Series | None = None,
    trend_efficiency_window: int | None = None,
    trend_efficiency_threshold_lookback: int = 504,
    trend_efficiency_boost: float = 1.5,
    cost_bps: float = DEFAULT_COST_BPS,
    construction: str = "held_contract",
    returns: dict[str, pd.Series] | None = None,
) -> dict:
    """Equal-weighted diversified TSMOM strategy and passive-long benchmark.

    Each instrument uses its own full available history; the portfolio on any given day is
    an equal-weighted average over the instruments that are live then (see
    `portfolio_mean`), so the effective universe grows as instruments' histories begin --
    matching how a real diversified futures book is built up.

    Returns are net of `cost_bps` basis points per unit of notional traded, applied
    identically to both legs (the trend book turns over far more than the benchmark, so a
    gross comparison flatters it). Pass `cost_bps=0.0` for gross.

    Two independent, optional regime layers, neither touching the passive benchmark
    (which by construction isn't a trend strategy and has no regime to be "aware" of):
    - `regime_scale` (see `correlation_regime_scale`): one shared, cross-sectional series
      multiplied into every instrument's position.
    - `trend_efficiency_window` (see `strategies.trend_efficiency_scale`): computed per
      instrument from that instrument's own returns.
    """
    returns = returns if returns is not None else instrument_returns(symbols, construction)

    strategy_cols, benchmark_cols = {}, {}
    strategy_positions, benchmark_positions = {}, {}
    for sym, r in returns.items():
        position = tsmom_position(r, lookback=lookback, vol_window=vol_window, target_vol=target_vol)
        if regime_scale is not None:
            position = position * regime_scale.reindex(position.index)
        if trend_efficiency_window is not None:
            position = position * trend_efficiency_scale(
                r,
                window=trend_efficiency_window,
                threshold_lookback=trend_efficiency_threshold_lookback,
                boost=trend_efficiency_boost,
            ).reindex(position.index)
        benchmark_position = passive_long_position(r, vol_window=vol_window, target_vol=target_vol)

        strategy_positions[sym] = position
        benchmark_positions[sym] = benchmark_position
        strategy_cols[sym] = (position * r).dropna()
        benchmark_cols[sym] = (benchmark_position * r).dropna()

    per_instrument_strategy = pd.DataFrame(strategy_cols)
    per_instrument_benchmark = pd.DataFrame(benchmark_cols)

    strategy = portfolio_mean(per_instrument_strategy)
    benchmark = portfolio_mean(per_instrument_benchmark)

    if cost_bps:
        # turnover is per unit of capital, so scale by the same 1/N the returns get
        strategy_turnover = portfolio_mean(pd.DataFrame(strategy_positions).diff().abs())
        benchmark_turnover = portfolio_mean(pd.DataFrame(benchmark_positions).diff().abs())
        strategy = apply_costs(strategy, strategy_turnover, cost_bps)
        benchmark = apply_costs(benchmark, benchmark_turnover, cost_bps)

    return {
        "strategy": strategy,
        "benchmark": benchmark,
        "per_instrument_strategy": per_instrument_strategy,
        "per_instrument_benchmark": per_instrument_benchmark,
        "strategy_positions": pd.DataFrame(strategy_positions),
        "benchmark_positions": pd.DataFrame(benchmark_positions),
    }


def correlation_regime_scale(
    symbols: list[str],
    corr_window: int = 120,
    threshold_lookback: int = 504,
    low_corr_scale: float = 1.0,
    high_corr_scale: float = 0.5,
    returns: dict[str, pd.Series] | None = None,
) -> pd.Series:
    """TSMOM position multiplier based on cross-market correlation regime.

    Per Hurst-Ooi-Pedersen: trend-following works better when markets move independently
    (low average correlation) and worse in "risk-on/risk-off" regimes (high average
    correlation). Full exposure (`low_corr_scale`, default 1.0) when today's correlation
    is below its own trailing median; half exposure (`high_corr_scale`, default 0.5) when
    above -- deliberately a cut, not a boost above baseline, to keep this a risk-reduction
    dial rather than adding leverage on top of an already-uncertain regime read.

    The trailing median (not a fixed threshold) is self-calibrating -- it uses only each
    day's own past correlation history -- and the whole scale is shifted by one day, same
    no-look-ahead discipline as `strategies.tsmom_position`. Before enough history exists
    to form either the correlation or its median the scale is NaN, not a tradeable
    default: an earlier version silently ran the book at half exposure for ~417 warm-up
    days because `NaN < threshold` evaluates to False.
    """
    returns = returns if returns is not None else instrument_returns(symbols)
    # correlation is scale-invariant, so raw returns go straight in -- no vol
    # normalisation, and hence no way to reintroduce the scaling bug it used to carry
    corr = average_pairwise_correlation(pd.DataFrame(returns), window=corr_window)
    threshold = corr.rolling(threshold_lookback, min_periods=corr_window).median()

    known = corr.notna() & threshold.notna()
    scale = pd.Series(np.nan, index=corr.index, dtype=float)
    scale[known] = np.where(corr[known] < threshold[known], low_corr_scale, high_corr_scale)
    return scale.shift(1)


def walk_forward(
    start_year: int = 2000,
    end_year: int = 2014,
    min_history_years: int = 5,
    **kwargs,
) -> dict:
    """Rolling walk-forward: re-select the universe each year-end, trade the next year OOS.

    At each year boundary the universe is re-derived point-in-time
    (`data.curated_futures_universe(as_of=...)`), so the markets traded in year Y are only
    those a backtester standing on 31-Dec-(Y-1) could have chosen. Each year's returns are
    then concatenated into a single out-of-sample track record.

    This is the brief's recommended "rolling walk forward analysis to conserve data". Note
    that TSMOM's parameters are a priori from the literature rather than fitted, so what
    this actually removes is universe-selection look-ahead, which is the binding source of
    in-sample contamination in this project.
    """
    strategy_years, benchmark_years, universes = [], [], {}
    for year in range(start_year, end_year + 1):
        as_of = pd.Timestamp(f"{year - 1}-12-31")
        universe = data.curated_futures_universe(as_of=as_of)
        if not universe:
            continue
        universes[year] = universe

        result = diversified_tsmom(universe, **kwargs)
        window = slice(f"{year}-01-01", f"{year}-12-31")
        strategy_years.append(result["strategy"].loc[window])
        benchmark_years.append(result["benchmark"].loc[window])

    return {
        "strategy": pd.concat(strategy_years).sort_index(),
        "benchmark": pd.concat(benchmark_years).sort_index(),
        "universes": universes,
    }
