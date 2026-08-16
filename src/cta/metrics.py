"""Performance metrics: Sharpe, drawdown, turnover/costs, growth-of-$1 helpers.

Also holds the inference machinery (`block_bootstrap` and friends). A backtest Sharpe is
a point estimate from one path of a highly autocorrelated series; quoting it without an
interval invites reading a difference of 0.15 as a result when it is noise. Everything
here is deliberately non-parametric -- daily returns are fat-tailed and serially
dependent, so the textbook Sharpe standard error (`sqrt((1 + S^2/2)/T)`) understates the
true spread.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from cta.indicators import TRADING_DAYS_PER_YEAR

# Resampling in blocks of ~one trading month preserves the volatility clustering and
# short-horizon autocorrelation that an i.i.d. bootstrap would destroy (and which, if
# destroyed, makes every interval look far too tight).
DEFAULT_BLOCK_DAYS = 21
DEFAULT_RESAMPLES = 2000


def sharpe_ratio(returns: pd.Series) -> float:
    """Annualized Sharpe ratio of a daily return series (no risk-free adjustment).

    No risk-free subtraction because every return series in this project is a *futures*
    return: a fully-collateralised futures position earns the cash rate plus the price
    return, so the price return already IS the excess return. Subtracting a risk-free
    rate again would double-count it.
    """
    return returns.mean() / returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def t_statistic(returns: pd.Series) -> float:
    """t-statistic of the mean daily return against zero.

    Numerically this is just `sharpe * sqrt(years)`, which is the useful intuition: five
    years of a Sharpe-1.1 strategy gives t ~ 2.5, whereas one year of the same strategy
    gives t ~ 1.1 and proves nothing. Reported alongside every headline Sharpe.
    """
    clean = returns.dropna()
    return float(clean.mean() / clean.std() * np.sqrt(len(clean)))


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


# --------------------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------------------


def block_bootstrap(
    frame: pd.DataFrame | pd.Series,
    statistic: Callable[[pd.DataFrame], float],
    n_resamples: int = DEFAULT_RESAMPLES,
    block: int = DEFAULT_BLOCK_DAYS,
    seed: int = 0,
) -> np.ndarray:
    """Circular block bootstrap of `statistic` over the rows of `frame`.

    Resamples contiguous blocks of `block` rows (with wrap-around) until the resample is
    as long as the original, then evaluates `statistic` on it. Returns the raw
    distribution so callers can take whatever quantiles they need.

    Two design choices worth defending:

    - **Blocks, not individual days.** Daily returns are serially dependent (volatility
      clusters, trends persist). An i.i.d. bootstrap breaks that dependence and produces
      confidence intervals that are far too narrow -- it would "prove" results that are
      not there.
    - **Circular (wrap-around), not plain moving-block.** In a plain moving-block
      bootstrap the first and last `block` observations can appear in fewer blocks than
      the middle of the sample, so they are systematically under-weighted. Wrapping makes
      every observation equally likely, which matters here because the most extreme
      observations in the sample (2008) sit nowhere near the middle.

    `frame` is resampled as a unit, so paired series (strategy and benchmark on the same
    days) stay aligned and the difference statistic keeps its correlation structure.
    """
    data = frame.to_frame() if isinstance(frame, pd.Series) else frame
    data = data.dropna()
    n_rows = len(data)
    if n_rows < block:
        raise ValueError(f"need at least {block} rows to bootstrap, got {n_rows}")

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n_rows / block))
    offsets = np.arange(block)

    out = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        starts = rng.integers(0, n_rows, n_blocks)
        # wrap with a modulo so blocks that run off the end continue from the start
        positions = (starts[:, None] + offsets[None, :]).ravel()[:n_rows] % n_rows
        out[i] = statistic(data.iloc[positions])
    return out


def _interval(draws: np.ndarray, point: float, confidence: float) -> dict:
    tail = (1.0 - confidence) / 2.0
    return {
        "estimate": float(point),
        "ci_low": float(np.percentile(draws, 100 * tail)),
        "ci_high": float(np.percentile(draws, 100 * (1 - tail))),
        "p_positive": float(np.mean(draws > 0)),
    }


def sharpe_interval(
    returns: pd.Series,
    n_resamples: int = DEFAULT_RESAMPLES,
    block: int = DEFAULT_BLOCK_DAYS,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """Bootstrap confidence interval for a single series' Sharpe ratio.

    Returns `estimate`, `ci_low`, `ci_high`, `p_positive` (the bootstrap probability that
    the true Sharpe exceeds zero) and `t_stat`.
    """
    clean = returns.dropna()
    draws = block_bootstrap(
        clean.to_frame("r"),
        lambda d: sharpe_ratio(d["r"]),
        n_resamples=n_resamples,
        block=block,
        seed=seed,
    )
    return {**_interval(draws, sharpe_ratio(clean), confidence), "t_stat": t_statistic(clean)}


def sharpe_difference_interval(
    strategy: pd.Series,
    benchmark: pd.Series,
    n_resamples: int = DEFAULT_RESAMPLES,
    block: int = DEFAULT_BLOCK_DAYS,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """Bootstrap confidence interval for the DIFFERENCE between two Sharpe ratios.

    Resamples both legs on the same days, so the (usually non-zero) correlation between
    them is preserved -- treating the two Sharpes as independent would overstate the
    precision of their difference.

    This is the statistic to check before claiming "A beats B". Two series can each have
    a comfortably significant Sharpe while the interval on their difference straddles
    zero, which is exactly the situation in this project: the trend book's own Sharpe is
    significant, its margin over the risk-matched control is not.
    """
    paired = pd.DataFrame({"s": strategy, "b": benchmark}).dropna()
    draws = block_bootstrap(
        paired,
        lambda d: sharpe_ratio(d["s"]) - sharpe_ratio(d["b"]),
        n_resamples=n_resamples,
        block=block,
        seed=seed,
    )
    point = sharpe_ratio(paired["s"]) - sharpe_ratio(paired["b"])
    return {**_interval(draws, point, confidence), "correlation": float(paired["s"].corr(paired["b"]))}


def rolling_sharpe(returns: pd.Series, window: int = TRADING_DAYS_PER_YEAR) -> pd.Series:
    """Trailing annualized Sharpe over a rolling window -- shows stability, not just level.

    A single full-sample Sharpe cannot distinguish a strategy that earned it evenly from
    one that earned it all in a single year, which is a distinction this project's own
    equity curve turns on.
    """
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std()
    return (mean / std * np.sqrt(TRADING_DAYS_PER_YEAR)).dropna()
