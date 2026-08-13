"""Signal building blocks: moving averages, vol estimators, z-scores, regime classifiers."""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def daily_returns(close: pd.Series) -> pd.Series:
    """Simple daily % returns of a price series.

    WARNING: correct for a genuine single-asset price series (a spot index, an ETF), but
    NOT for a spliced futures continuous contract -- see `naive_spliced_returns`. Use
    `held_contract_returns` for futures.
    """
    return close.pct_change().dropna()


def held_contract_returns(back_adjusted_close: pd.Series, raw_close: pd.Series) -> pd.Series:
    """Return of actually *holding* a futures contract, excluding the roll gap.

    This is the Moskowitz-Ooi-Pedersen / Baltas-Kosowski convention: compute the return
    while holding a single contract, and splice the *return* series across rolls -- never
    take a percentage change across a roll-day price jump.

    Mechanically: the back-adjusted (`_CCB`) series differs from the raw series by a
    per-roll constant offset, so its first difference is the true same-contract price
    change on every day (verified: `raw.diff()` and `ccb.diff()` agree to floating-point
    tolerance on every non-roll day, and differ only on `Delivery Month` changes). Dividing
    that difference by the *raw* previous close gives a percentage return.

    Using the raw close as the denominator is what makes this safe: the `_CCB` levels go
    negative for 14 of 94 markets (which is why an earlier version of this project used raw
    prices throughout), but that affects levels, not differences -- and raw closes are never
    non-positive.
    """
    idx = back_adjusted_close.index.intersection(raw_close.index)
    return (back_adjusted_close.loc[idx].diff() / raw_close.loc[idx].shift(1)).dropna()


def naive_spliced_returns(raw_close: pd.Series) -> pd.Series:
    """INCORRECT for futures -- retained solely to demonstrate the bug it causes.

    `pct_change()` straight off a raw spliced continuous contract books the calendar
    spread between the expiring and next contract as if it were tradeable P&L. The
    injected return is systematically signed by contango/backwardation, so an always-long
    book harvests it while a long/short book pays it.

    How wrong: over 2005-2014 this construction implies a rolling long VIX futures position
    returned +30.3% (true: -99.9%) and long WTI +26.5% (true: -58.3%). See
    notebooks/02_strategy_research.ipynb, where this function reproduces the original
    contaminated result side by side with `held_contract_returns`.
    """
    return raw_close.pct_change().dropna()


def realized_vol(returns: pd.Series, window: int = 60) -> pd.Series:
    """Annualized realized volatility: rolling std of daily returns, scaled by sqrt(252).

    Simpler than MOP's exponentially-weighted estimator (see research/literature_notes.md)
    -- a documented, optional refinement, not required for a baseline.
    """
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def momentum_signal(returns: pd.Series, lookback: int = 252) -> pd.Series:
    """Sign of the trailing `lookback`-day compounded return (Moskowitz-Ooi-Pedersen TSMOM).

    +1 if the instrument is up over the lookback window, -1 if down, 0 if exactly flat.
    """
    cum_return = (1 + returns).cumprod()
    trailing_return = cum_return / cum_return.shift(lookback) - 1
    return np.sign(trailing_return)


def average_pairwise_correlation(
    returns: pd.DataFrame,
    window: int = 120,
    step: int = 21,
    min_overlap: int = 30,
) -> pd.Series:
    """Rolling average pairwise correlation across instruments.

    Hurst-Ooi-Pedersen's "Century of Evidence" paper found average cross-market
    correlation is the strongest predictor of trend-following performance -- low
    correlation (markets moving independently) favors trend-following, high correlation
    ("risk-on/risk-off" regimes) hurts it (research/literature_notes.md).

    Computed exactly: the mean off-diagonal entry of a pairwise-complete correlation
    matrix over the trailing `window`, evaluated every `step` trading days and
    forward-filled between evaluations (~460 evaluations over this dataset). Stepping
    keeps an O(N^2) matrix affordable while staying exact at each evaluation, and it is
    far easier to explain than a variance identity -- which matters for the brief's
    "explainable" criterion.

    Correlation is scale-invariant, so this takes RAW returns: no volatility
    normalisation, and therefore no way to reintroduce the scaling bug that a previous
    identity-based implementation had (it silently returned a constant -1/(N-1) for the
    entire sample because it was fed annualised-vol-normalised inputs whose standard
    deviation was 1/sqrt(252), not 1).

    Pairs with fewer than `min_overlap` overlapping observations in the window are
    excluded, so instruments whose histories barely overlap cannot dominate the estimate.
    """
    n_cols = returns.shape[1]
    if n_cols < 2:
        return pd.Series(np.nan, index=returns.index)

    values = pd.Series(np.nan, index=returns.index, dtype=float)
    for i in range(window - 1, len(returns), step):
        block = returns.iloc[i - window + 1 : i + 1]
        # a pair needs enough joint observations for its correlation to mean anything
        counts = block.notna().astype(float)
        overlap = counts.T @ counts
        corr = block.corr(min_periods=min_overlap).to_numpy(dtype=float)
        mask = overlap.to_numpy(dtype=float) >= min_overlap
        np.fill_diagonal(mask, False)
        valid = mask & np.isfinite(corr)
        if valid.sum() > 0:
            values.iloc[i] = float(corr[valid].mean())

    return values.ffill().clip(-1, 1)


def efficiency_ratio(returns: pd.Series, window: int = 252) -> pd.Series:
    """Kaufman's Efficiency Ratio: |net move| / (sum of |daily moves|) over the trailing
    window, on the price path implied by `returns`. Bounded in [0, 1] -- close to 1 means
    a smooth, persistent, low-reversal trend (net move ~= total distance travelled);
    close to 0 means choppy, back-and-forth movement that mostly cancels out.

    This targets a different, more specific thing than `average_pairwise_correlation`:
    a single instrument's own trend can be highly efficient while unrelated to what other
    markets are doing. See notebooks/02_strategy_research.ipynb -- vol-scaled position
    sizing shrinks a position whenever volatility rises, even if the trend itself is
    intact. Tested in notebooks/02_strategy_research.ipynb and found NOT to improve
    risk-adjusted performance -- retained because the negative result is documented
    there, not because it earns a place in the strategy.
    """
    price = (1 + returns).cumprod()
    net_move = (price - price.shift(window)).abs()
    path_length = price.diff().abs().rolling(window).sum()
    return net_move / path_length
