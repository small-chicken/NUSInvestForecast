"""Performance metrics: Sharpe, drawdown, turnover/costs, growth-of-$1 helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cta.indicators import TRADING_DAYS_PER_YEAR


def sharpe_ratio(returns: pd.Series) -> float:
    """Annualized Sharpe ratio of a daily return series (no risk-free adjustment)."""
    return returns.mean() / returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def growth_of_one(returns: pd.Series) -> pd.Series:
    """Cumulative growth of $1 invested at the start of the series."""
    return (1 + returns).cumprod()


def cumulative_return(returns: pd.Series) -> float:
    """Total compounded return over the series."""
    return growth_of_one(returns).iloc[-1] - 1


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough decline in cumulative wealth, floored at -100%.

    On a levered series the compounded wealth path can cross zero, after which
    `wealth / wealth.cummax() - 1` is meaningless (it produced values like -284% on some
    single-instrument legs). Once wealth hits zero the account is wiped out, so the
    drawdown is -100% and nothing worse is representable.
    """
    wealth = growth_of_one(returns)
    if (wealth <= 0).any():
        return -1.0
    return float((wealth / wealth.cummax() - 1).min())


def annualized_volatility(returns: pd.Series) -> float:
    """Annualized standard deviation of a daily return series."""
    return float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def turnover(positions: pd.DataFrame | pd.Series) -> pd.Series:
    """Daily gross turnover: total absolute change in position across instruments.

    Units are "notional traded per unit of capital", matching how positions are sized
    (`target_vol / realized_vol`), so multiplying by a per-unit cost gives a return drag.
    """
    changes = positions.diff().abs()
    return changes.sum(axis=1) if isinstance(changes, pd.DataFrame) else changes


def apply_costs(returns: pd.Series, positions: pd.DataFrame | pd.Series, cost_bps: float) -> pd.Series:
    """Subtract `cost_bps` basis points per unit of notional traded from `returns`.

    The trend book turns over roughly 58x its capital per year against the passive
    benchmark's 9x, so comparing the two gross flatters the strategy -- costs are applied
    to both legs on the same basis.
    """
    drag = turnover(positions).reindex(returns.index).fillna(0.0) * (cost_bps / 1e4)
    return returns - drag
