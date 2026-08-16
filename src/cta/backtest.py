"""Combine per-instrument positions into a diversified portfolio backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cta import data
from cta.indicators import (
    TRADING_DAYS_PER_YEAR,
    average_pairwise_correlation,
    held_contract_returns,
    naive_spliced_returns,
)
from cta.metrics import apply_costs, max_drawdown, sharpe_ratio
from cta.strategies import passive_long_position, trend_efficiency_scale, tsmom_position

DEFAULT_COST_BPS = 2.0

# The book-level volatility target used whenever portfolio vol targeting is switched on.
# 10% annualized is Hurst-Ooi-Pedersen's own choice and is roughly what a diversified
# managed-futures programme is sold at; it is a units convention, not a tuned parameter --
# Sharpe is invariant to a *constant* rescaling, so nothing here is fitted by picking 10%.
DEFAULT_PORTFOLIO_VOL_TARGET = 0.10


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


def portfolio_mean(per_instrument: pd.DataFrame, delist_after: int = 21) -> pd.Series:
    """Equal-weight across LIVE instruments, booking closed markets at 0 P&L.

    A plain `.mean(axis=1)` uses pandas' default skipna=True, which drops a market that is
    shut for a local holiday from the DIVISOR rather than crediting it 0 P&L -- so every
    open market's weight jumps from 1/N_live to 1/N_open. In 2005-2014 that silently
    re-levered the book on 540 partial days, and on 7 of them booked a "77-market
    diversified portfolio" as 3 markets at a 40% vol target each.

    An instrument is "live" from its first observation until it has been silent for
    `delist_after` consecutive trading days. Inside that span a missing value means the
    exchange was closed, so the position simply earns nothing. Before it, the instrument
    does not exist yet and is correctly absent from the denominator -- which is what lets
    the universe ramp up over time.

    The `delist_after` rule replaces an earlier `valid[::-1].cummax()[::-1]`, which marked
    an instrument dead from its last observation onward. That was a (small) look-ahead:
    reverse-cumulative-max asks "is there any observation at or after today?", which a
    backtester standing on that day cannot know. A trailing silence rule is the same idea
    using only the past -- you notice a market has delisted by observing that it stopped
    printing, which takes some days. Over this dataset the two agree almost everywhere
    (every series ends on the same date), so this is discipline rather than a fix.
    """
    valid = per_instrument.notna()
    started = valid.cummax()

    # index position of each column's most recent valid observation, forward-filled
    row_number = pd.DataFrame(
        np.repeat(np.arange(len(per_instrument))[:, None], per_instrument.shape[1], axis=1),
        index=per_instrument.index,
        columns=per_instrument.columns,
    )
    days_since_last_print = row_number - row_number.where(valid).ffill()
    live = started & (days_since_last_print <= delist_after)

    n_live = live.sum(axis=1)
    contributions = per_instrument.where(valid, 0.0).where(live, 0.0)
    return (contributions.sum(axis=1) / n_live.replace(0, np.nan)).dropna()


def portfolio_leverage(
    returns: pd.Series,
    target_vol: float,
    window: int = 252,
    max_leverage: float = 3.0,
) -> pd.Series:
    """Book-level leverage that holds the *portfolio's* realized volatility near a target.

    `tsmom_position` already targets volatility per market, but that does not pin down the
    volatility of the combined book: the portfolio's vol also depends on how correlated
    the markets happen to be and on how many are live, both of which move a great deal
    over a 36-year sample. In this project the trend book's trailing realized vol ranges
    from 7.3% (1993) to 17.5% (2008) despite every individual position being sized to the
    same 40% target -- so risk more than doubles precisely when markets are most dangerous.

    Hurst-Ooi-Pedersen add exactly this layer on top of MOP's per-market sizing. Scaling
    by `target_vol / trailing_realized_vol` de-levers the book after volatility has
    already risen and re-levers it once conditions calm down.

    Shifted one day, so the leverage applied on day t is computed from returns through
    t-1. Capped at `max_leverage` for the same reason position sizing is capped: a quiet
    stretch should not be able to lever the whole book without bound.
    """
    realized = returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    return (target_vol / realized).clip(upper=max_leverage).shift(1)


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
    portfolio_vol_target: float | None = None,
    portfolio_vol_window: int = 252,
    max_portfolio_leverage: float = 3.0,
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

    `portfolio_vol_target` adds a book-level volatility target on top of the per-market
    one (see `portfolio_leverage`). Unlike the regime layers it IS applied to the
    benchmark as well, using the benchmark's own realized vol: the benchmark exists to
    hold risk constant while the signal varies, so an overlay that changes the risk
    profile of one leg but not the other would break exactly the comparison it is for.
    """
    returns = returns if returns is not None else instrument_returns(symbols, construction)

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

        strategy_positions[sym] = position
        benchmark_positions[sym] = passive_long_position(
            r, vol_window=vol_window, target_vol=target_vol
        )

    strategy = _assemble_book(
        strategy_positions, returns, cost_bps,
        portfolio_vol_target, portfolio_vol_window, max_portfolio_leverage,
    )
    benchmark = _assemble_book(
        benchmark_positions, returns, cost_bps,
        portfolio_vol_target, portfolio_vol_window, max_portfolio_leverage,
    )

    return {
        "strategy": strategy["returns"],
        "benchmark": benchmark["returns"],
        "per_instrument_strategy": strategy["per_instrument"],
        "per_instrument_benchmark": benchmark["per_instrument"],
        "strategy_positions": strategy["positions"],
        "benchmark_positions": benchmark["positions"],
        "strategy_leverage": strategy["leverage"],
        "benchmark_leverage": benchmark["leverage"],
    }


def _assemble_book(
    positions: dict[str, pd.Series],
    returns: dict[str, pd.Series],
    cost_bps: float,
    portfolio_vol_target: float | None,
    portfolio_vol_window: int,
    max_portfolio_leverage: float,
) -> dict:
    """Turn per-instrument positions into a net-of-cost portfolio return series.

    The book-level volatility target is applied here rather than to the finished return
    series, and the ordering is what makes it honest:

    1. Combine the *unlevered* positions into a portfolio return.
    2. Derive leverage from that series' trailing realized vol, lagged one day. Using the
       unlevered book keeps this non-circular -- leverage on day t depends only on returns
       through t-1, which are themselves not a function of today's leverage.
    3. Re-scale each instrument's POSITION by that leverage, then recompute turnover.

    Step 3 is the part it would be easy to get wrong. Scaling the finished return series
    would capture the P&L effect of leverage while silently ignoring the trades needed to
    change leverage -- so de-levering into a crisis would appear free. Scaling positions
    first means every re-levering trade is charged at the same cost per unit notional as
    any other trade.
    """
    unlevered_pnl = pd.DataFrame({s: (positions[s] * returns[s]).dropna() for s in positions})
    unlevered = portfolio_mean(unlevered_pnl)

    if portfolio_vol_target is None:
        leverage = None
        position_frame = pd.DataFrame(positions)
        per_instrument, gross = unlevered_pnl, unlevered
    else:
        leverage = portfolio_leverage(
            unlevered,
            target_vol=portfolio_vol_target,
            window=portfolio_vol_window,
            max_leverage=max_portfolio_leverage,
        )
        position_frame = pd.DataFrame(positions).mul(leverage, axis=0)
        per_instrument = pd.DataFrame(
            {s: (position_frame[s] * returns[s]).dropna() for s in positions}
        )
        gross = portfolio_mean(per_instrument)

    net = gross
    if cost_bps:
        # turnover is per unit of capital, so scale by the same 1/N the returns get
        turnover = portfolio_mean(position_frame.diff().abs())
        net = apply_costs(gross, turnover, cost_bps)

    return {
        "returns": net,
        "per_instrument": per_instrument,
        "positions": position_frame,
        "leverage": leverage,
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


# --------------------------------------------------------------------------------------
# Reference books
# --------------------------------------------------------------------------------------

# Liquid, long-history proxies for the two things an allocator already owns.
EQUITY_SYMBOL = "&ES"  # E-mini S&P 500
BOND_SYMBOL = "&ZN"  # 10-year US Treasury note


def reference_books(returns: dict[str, pd.Series] | None = None) -> dict[str, pd.Series]:
    """Investable reference portfolios, for context the risk-matched control cannot give.

    `diversified_tsmom`'s "passive long" benchmark is the right *scientific* control -- it
    holds universe and risk-targeting constant so the only difference is the timing signal
    -- but it is not a thing anyone can buy: a 40%-per-market always-long futures book
    draws down 77%. Beating it answers "does the trend signal add value?", not "should I
    allocate to this?".

    These two answer the second question:
    - **Equity**: unlevered long E-mini S&P 500 futures.
    - **60/40**: 60% equity futures, 40% 10-year Treasury note futures, rebalanced daily.

    Both are futures returns, so like every other series in this project they are already
    *excess* returns over cash and are directly comparable to the strategy without any
    risk-free adjustment. They are deliberately unlevered: the point is to show what the
    trend book adds to a normal portfolio, at each one's natural risk.
    """
    returns = returns if returns is not None else instrument_returns([EQUITY_SYMBOL, BOND_SYMBOL])
    equity = returns[EQUITY_SYMBOL]
    bonds = returns[BOND_SYMBOL]
    blended = (0.6 * equity).add(0.4 * bonds, fill_value=0.0).loc[bonds.index.min() :]
    return {"Equity (S&P futures)": equity, "60/40 equity/bonds": blended}


def blend(core: pd.Series, sleeve: pd.Series, sleeve_weight: float) -> pd.Series:
    """Daily-rebalanced mix of a core portfolio and a satellite sleeve.

    The allocator's question, and the one the risk-matched control cannot answer: a
    strategy with a near-zero correlation to equities does not have to *beat* a 60/40
    portfolio to be worth owning, it only has to improve it. Mixing at a realistic sleeve
    weight tests that directly.

    Evaluated on the days both series are available, so the comparison is like-for-like.
    """
    paired = pd.DataFrame({"core": core, "sleeve": sleeve}).dropna()
    return (1 - sleeve_weight) * paired["core"] + sleeve_weight * paired["sleeve"]


# --------------------------------------------------------------------------------------
# Universe studies
# --------------------------------------------------------------------------------------


def _subset_book(
    positions: dict[str, pd.Series],
    returns: dict[str, pd.Series],
    subset: list[str],
    cost_bps: float,
    portfolio_vol_target: float | None = None,
) -> pd.Series:
    """Portfolio return for a subset of already-computed per-instrument positions.

    Positions are per-instrument and independent of which other markets are in the book,
    so a breadth study can compute them once for the full universe and re-combine subsets
    -- which is what makes hundreds of random draws affordable.

    The book-level vol target is re-derived per subset rather than reused, because it is a
    property of the *portfolio*: a five-market book is far more volatile than a 77-market
    one at the same per-market sizing, so they need different leverage to hit the same
    target. Reusing the full universe's leverage would confound breadth with risk level --
    which is the one thing this study exists to separate.
    """
    return _assemble_book(
        {s: positions[s] for s in subset},
        returns,
        cost_bps,
        portfolio_vol_target,
        portfolio_vol_window=252,
        max_portfolio_leverage=3.0,
    )["returns"]


def universe_breadth_study(
    symbols: list[str],
    sizes: list[int],
    window: slice,
    n_draws: int = 15,
    seed: int = 0,
    cost_bps: float = DEFAULT_COST_BPS,
    returns: dict[str, pd.Series] | None = None,
    portfolio_vol_target: float | None = None,
    **position_kwargs,
) -> pd.DataFrame:
    """Sharpe of the TSMOM book as a function of how many markets it trades.

    For each size in `sizes`, draw `n_draws` random subsets without replacement and
    evaluate each over `window`. Random subsets, not "the best N": any rule for picking
    which markets to keep would itself be a decision made with hindsight, and the question
    here is what breadth alone is worth.

    This is the direct test of the most common suggestion made about a diversified trend
    book -- "trade fewer, better markets" -- and the answer on this dataset is that
    breadth is not a detail of the strategy, it *is* the strategy.
    """
    returns = returns if returns is not None else instrument_returns(symbols)
    positions = {
        s: tsmom_position(r, **position_kwargs) for s, r in returns.items() if s in set(symbols)
    }

    rng = np.random.default_rng(seed)
    rows = []
    for size in sizes:
        # a "draw" of the full universe is the universe; one deterministic row, not n
        draws = 1 if size >= len(symbols) else n_draws
        for draw in range(draws):
            subset = (
                list(symbols)
                if size >= len(symbols)
                else list(rng.choice(symbols, size=size, replace=False))
            )
            book = _subset_book(
                positions, returns, subset, cost_bps, portfolio_vol_target
            ).loc[window]
            rows.append(
                {
                    "n_markets": min(size, len(symbols)),
                    "draw": draw,
                    "sharpe": sharpe_ratio(book),
                    "max_drawdown": max_drawdown(book),
                }
            )
    return pd.DataFrame(rows)


def correlation_ranked_universe(
    symbols: list[str],
    n: int,
    as_of: str | pd.Timestamp,
    most_correlated: bool = True,
    returns: dict[str, pd.Series] | None = None,
    min_overlap: int = 250,
) -> list[str]:
    """The `n` most (or least) mutually correlated markets, ranked POINT-IN-TIME.

    Each market is scored by its average correlation to every other market in `symbols`,
    computed using only returns up to `as_of`. Ranking on the full sample and then
    evaluating out-of-sample would be look-ahead -- the selection would already know which
    markets turned out to diversify -- so the cut-off matters more here than anywhere else
    in the project.

    Used to test the "concentrate the book into a few closely-related markets" idea
    against its opposite, on equal terms.
    """
    returns = returns if returns is not None else instrument_returns(symbols)
    panel = pd.DataFrame({s: returns[s] for s in symbols}).loc[: pd.Timestamp(as_of)]

    corr = panel.corr(min_periods=min_overlap)
    # drop the diagonal: every market correlates 1.0 with itself, which would otherwise
    # pull each score toward 1/N and compress the ranking
    average_correlation = corr.mask(np.eye(len(corr), dtype=bool)).mean(skipna=True).dropna()

    ordered = average_correlation.sort_values(ascending=not most_correlated)
    return sorted(ordered.head(n).index)
