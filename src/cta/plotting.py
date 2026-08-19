"""Shared chart styling helpers used across notebooks.

Colours come from a validated categorical palette (checked for colour-vision
deficiency separation, chroma, lightness band and contrast against the chart
surface). Slot 1 (blue) is always the strategy, slot 2 (orange) always the
benchmark -- colour follows the entity, never its rank, so a chart that drops a
series never repaints the survivors.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# Categorical slots, in fixed order. Never cycle past these; fold a tail into
# "Other" or facet into small multiples instead.
STRATEGY = "#2a78d6"  # slot 1, blue
BENCHMARK = "#eb6834"  # slot 2, orange
ACCENT = "#1baf7a"  # slot 3, aqua
REFERENCE = "#1baf7a"  # slot 3, aqua -- 60/40, the investable reference book
REFERENCE_ALT = "#eda100"  # slot 4, yellow -- equity, the second reference book

# Colour is bound to the ENTITY, not to its position in a chart, so a series keeps its
# colour across every exhibit in the notebook and a chart that drops a series never
# repaints the survivors. Slots 3 and 4 sit below 3:1 contrast on this surface, which
# under the palette's relief rule obliges visible direct labels -- every chart below that
# uses them calls `annotate_last`.
SERIES_COLOURS = {
    "TSMOM": STRATEGY,
    "Passive long": BENCHMARK,
    "60/40 equity/bonds": REFERENCE,
    "Equity (S&P futures)": REFERENCE_ALT,
}

# A variant of an entity that has been superseded (e.g. the book before an overlay is
# added) is drawn in neutral ink rather than a second hue: it is the same entity, and
# giving it a categorical colour would falsely imply a separate one.
SUPERSEDED = "#9a9892"

# Diverging pair for signed quantities (warm/cool poles, neutral gray midpoint).
POSITIVE = "#2a78d6"
NEGATIVE = "#d03b3b"
MIDPOINT = "#f0efec"

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2df"


def use_house_style() -> None:
    """Apply the shared matplotlib style: thin marks, recessive chrome, ink text."""
    mpl.rcParams.update(
        {
            "figure.figsize": (9, 4.5),
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titlesize": 12,
            "axes.titleweight": "demibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            # solid hairline grid -- dashed grids read as noise
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "text.color": TEXT_PRIMARY,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": "tight",
            "savefig.dpi": 150,
        }
    )


def annotate_last(ax, series: pd.Series, label: str, color: str) -> None:
    """Direct-label the end of a line, so identity is never colour-alone."""
    ax.annotate(
        f"  {label}",
        xy=(series.index[-1], series.iloc[-1]),
        xytext=(4, 0),
        textcoords="offset points",
        color=color,
        fontsize=9,
        fontweight="demibold",
        va="center",
    )


def growth_chart(
    strategy: pd.Series,
    benchmark: pd.Series,
    title: str,
    ax=None,
    labels: tuple[str, str] = ("TSMOM", "Passive long"),
):
    """Growth of $1, log scale -- the standard trend-following exhibit.

    Log scale because a 30x cumulative return on a linear axis compresses the
    entire first two decades into a flat line.

    `labels` renames the two series where the same two-book layout is reused for a
    different pair. It is a parameter rather than something the caller fixes afterwards
    with `ax.legend([...])`, because re-labelling after the fact silently re-binds names
    to whatever handles matplotlib collected -- which on the drawdown chart below means
    the fills, not the lines, and produces a legend with the wrong colours.
    """
    ax = ax or plt.gca()
    ax.plot(strategy.index, strategy.to_numpy(), color=STRATEGY, label=labels[0])
    ax.plot(benchmark.index, benchmark.to_numpy(), color=BENCHMARK, label=labels[1])
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel("growth of $1 (log)")
    ax.legend(loc="upper left")
    return ax


def drawdown_chart(
    strategy: pd.Series,
    benchmark: pd.Series,
    title: str,
    ax=None,
    labels: tuple[str, str] = ("TSMOM", "Passive long"),
):
    """Underwater plot -- drawdown is what a Sharpe ratio hides."""
    ax = ax or plt.gca()
    handles = []
    for series, color, label in [(strategy, STRATEGY, labels[0]), (benchmark, BENCHMARK, labels[1])]:
        wealth = (1 + series).cumprod()
        underwater = wealth / wealth.cummax() - 1
        ax.fill_between(underwater.index, underwater.to_numpy(), 0, color=color, alpha=0.18, linewidth=0)
        line, = ax.plot(underwater.index, underwater.to_numpy(), color=color, label=label, linewidth=1.5)
        handles.append(line)
    ax.set_title(title)
    ax.set_ylabel("drawdown")
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    # legend built from the LINE handles only -- the fills are the same entities drawn
    # twice, and letting them into the legend duplicates every series
    ax.legend(handles=handles, loc="lower left")
    return ax


def diverging_barh(values: pd.Series, title: str, xlabel: str, ax=None):
    """Horizontal bars for a signed quantity: blue positive, red negative.

    Diverging rather than categorical because the sign *is* the message.
    """
    ax = ax or plt.gca()
    ordered = values.sort_values()
    colors = [POSITIVE if v >= 0 else NEGATIVE for v in ordered]
    ax.barh(ordered.index.astype(str), ordered.to_numpy(), color=colors, height=0.68)
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    # A bar's LENGTH is its value, so the axis must include zero even when every value has
    # the same sign -- starting it at the smallest value would make a 0.49 bar look like a
    # quarter of a 1.27 bar rather than 40% of it. Headroom is added beyond zero, never
    # taken out of it, so the end-of-bar labels still clear the tick labels.
    lower = min(0.0, ordered.min())
    upper = max(0.0, ordered.max())
    span = upper - lower
    ax.set_xlim(lower - 0.22 * span * (lower < 0), upper + 0.22 * span)
    for name, v in ordered.items():
        ax.annotate(
            f"{v:+.2f}",
            xy=(v, str(name)),
            xytext=(4 if v >= 0 else -4, 0),
            textcoords="offset points",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=9,
            color=TEXT_SECONDARY,
        )
    return ax


def multi_growth_chart(series: dict[str, pd.Series], title: str, ax=None):
    """Growth of $1 for several books at once, log scale, every line direct-labelled.

    Used where the point is a ranking among more than two portfolios (the strategy against
    the investable reference books). Colours come from `SERIES_COLOURS` so a book keeps
    its identity across exhibits; a legend is present *and* every line is labelled at its
    right-hand end, so identity is never carried by colour alone.

    Each series is re-based to $1 on the first day they all share -- otherwise a book with
    a longer history starts from a higher level and the chart compares start dates rather
    than performance.
    """
    ax = ax or plt.gca()
    start = max(s.index.min() for s in series.values())

    ends = []
    for name, returns in series.items():
        wealth = (1 + returns.loc[start:]).cumprod()
        colour = SERIES_COLOURS.get(name, STRATEGY)
        ax.plot(wealth.index, wealth.to_numpy(), color=colour, label=name)
        ends.append((name, colour, wealth.index[-1], wealth.iloc[-1]))

    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel("growth of $1 (log)")
    ax.legend(loc="upper left")
    # right-hand headroom for the direct labels, taken only on the right -- symmetric
    # margins would open dead space before the first observation as well
    ax.set_xlim(left=start, right=max(e[2] for e in ends) + (max(e[2] for e in ends) - start) * 0.17)
    label_ends_without_overlap(ax, ends)
    return ax


def label_ends_without_overlap(ax, ends: list[tuple], min_gap_points: float = 12.0) -> None:
    """Direct-label each line's end, nudging labels apart where two lines finish together.

    Books that end at a similar level would otherwise print their labels on top of each
    other -- which is exactly what happens here, since the whole point of the exhibit is
    that two of the reference books land in the same place. Positions are resolved in
    DISPLAY space (a log y-axis makes equal data offsets unequal on screen), so the figure
    is drawn once first to make the transform valid.
    """
    ax.get_figure().canvas.draw()
    placed = []
    for name, colour, x, y in sorted(ends, key=lambda e: e[3]):
        display_y = ax.transData.transform((0, y))[1]
        target = display_y
        if placed and target - placed[-1] < min_gap_points:
            target = placed[-1] + min_gap_points
        placed.append(target)
        ax.annotate(
            f"  {name}",
            xy=(x, y),
            xytext=(4, target - display_y),
            textcoords="offset points",
            color=colour,
            fontsize=9,
            fontweight="demibold",
            va="center",
        )


def interval_chart(rows: list[dict], title: str, xlabel: str, ax=None):
    """Forest plot: a point estimate and its confidence interval, one row per book.

    The exhibit that keeps a Sharpe ratio honest. A bar chart of point estimates invites
    reading a difference of 0.15 as a result; drawing the interval makes the reader see
    immediately whether it clears zero.

    Each row is a dict with `label`, `estimate`, `ci_low`, `ci_high`, and optionally
    `colour`. A reference line at zero is drawn because "is this interval clear of zero?"
    is the whole question.
    """
    ax = ax or plt.gca()
    positions = range(len(rows))
    for y, row in zip(positions, rows):
        colour = row.get("colour", STRATEGY)
        ax.plot(
            [row["ci_low"], row["ci_high"]], [y, y],
            color=colour, linewidth=2.0, solid_capstyle="round", alpha=0.55,
        )
        # 2px surface ring so the point reads clearly against its own interval line
        ax.plot(
            row["estimate"], y, "o", markersize=9, color=colour,
            markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=3,
        )
        ax.annotate(
            f"  {row['estimate']:+.2f}  [{row['ci_low']:+.2f}, {row['ci_high']:+.2f}]",
            xy=(row["ci_high"], y), xytext=(8, 0), textcoords="offset points",
            va="center", fontsize=9, color=TEXT_SECONDARY,
        )
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=1.0)
    ax.set_yticks(list(positions))
    ax.set_yticklabels([row["label"] for row in rows])
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    # right-hand headroom for the value labels, which are drawn in points and so do not
    # scale with the data range -- 45% of the span clears them at this figure width
    span = max(row["ci_high"] for row in rows) - min(row["ci_low"] for row in rows)
    ax.set_xlim(min(row["ci_low"] for row in rows) - 0.08 * span,
                max(row["ci_high"] for row in rows) + 0.45 * span)
    return ax


def bootstrap_density(draws, point: float, title: str, xlabel: str, ax=None):
    """Histogram of a bootstrap distribution, with zero and the point estimate marked.

    Single series, so no legend box -- the title names what is plotted. The mass to the
    left of zero is the number that matters and is annotated directly, because a reader
    should not have to integrate a histogram by eye.
    """
    ax = ax or plt.gca()
    below = float((draws <= 0).mean())
    ax.hist(draws, bins=48, color=STRATEGY, alpha=0.55, edgecolor=SURFACE, linewidth=0.8)
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=1.2)
    ax.axvline(point, color=NEGATIVE, linewidth=2.0)
    ax.annotate(
        f"observed {point:+.2f}", xy=(point, ax.get_ylim()[1]), xytext=(6, -12),
        textcoords="offset points", color=NEGATIVE, fontsize=9, fontweight="demibold",
    )
    ax.annotate(
        f"{below:.0%} of resamples\nfall at or below zero",
        xy=(0, ax.get_ylim()[1]), xytext=(-8, -12), textcoords="offset points",
        ha="right", va="top", color=TEXT_SECONDARY, fontsize=9,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("bootstrap resamples")
    ax.grid(axis="x", visible=False)
    return ax


def vol_target_chart(before: pd.Series, after: pd.Series, target: float, title: str, ax=None):
    """Trailing realized volatility of the book, with and without the book-level target.

    The 'before' line is the same entity in an earlier form, so it is drawn in neutral ink
    rather than a second categorical hue. The target is a dashed reference *annotation*,
    not a data series -- it is the only dashed mark in the notebook, and it is chrome.
    """
    ax = ax or plt.gca()
    ax.plot(before.index, before.to_numpy(), color=SUPERSEDED, linewidth=1.6,
            label="per-market vol target only")
    ax.plot(after.index, after.to_numpy(), color=STRATEGY, label="+ book-level vol target")
    ax.axhline(target, color=TEXT_SECONDARY, linewidth=1.0, linestyle="--")
    # label the target in the right-hand margin rather than over the series -- at 36 years
    # of daily data there is no interior whitespace left to put it in
    ax.margins(x=0.09)
    ax.annotate(
        f"{target:.0%} target", xy=(before.index[-1], target), xytext=(8, 0),
        textcoords="offset points", ha="left", va="center", color=TEXT_SECONDARY, fontsize=9,
        # knock the reference line out from behind the text rather than reading through it
        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5},
    )
    ax.set_title(title)
    ax.set_ylabel("trailing 1-year realized vol")
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.legend(loc="upper left")
    return ax


def breadth_chart(study: pd.DataFrame, title: str, ax=None):
    """Sharpe against number of markets traded: every random draw, plus the mean.

    Plotting the individual draws rather than only their mean is the point of the exhibit:
    it shows that a small book is not merely worse on average but wildly *dispersed* --
    which market you happen to pick starts to matter more than the strategy does.
    """
    ax = ax or plt.gca()
    ax.scatter(
        study["n_markets"], study["sharpe"], s=30, color=SUPERSEDED, alpha=0.55,
        edgecolor=SURFACE, linewidth=0.8, label="individual random draws", zorder=2,
    )
    means = study.groupby("n_markets")["sharpe"].mean()
    ax.plot(means.index, means.to_numpy(), color=STRATEGY, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2.0, label="mean across draws", zorder=3)
    ax.set_title(title)
    ax.set_xlabel("number of markets in the book")
    ax.set_ylabel("out-of-sample Sharpe")
    ax.legend(loc="lower right")
    return ax


def rolling_sharpe_chart(series: dict[str, pd.Series], title: str, window_label: str, ax=None):
    """Trailing Sharpe over time -- shows whether a headline number was earned evenly."""
    ax = ax or plt.gca()
    for name, values in series.items():
        colour = SERIES_COLOURS.get(name, STRATEGY)
        ax.plot(values.index, values.to_numpy(), color=colour, label=name)
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=1.0)
    ax.set_title(title)
    ax.set_ylabel(f"trailing {window_label} Sharpe")
    ax.legend(loc="lower left")
    return ax
