# NUS Investment Society QR Recruitment: Q3

Trading strategy (momentum/trend-following CTA, dynamic regime allocation, or vol
strategy) on G10 FX / Gold Continuous Futures / Govt Bond ETFs or Futures, with ≥5 years
of out-of-sample walk-forward results, benchmarked against literature.

## What was built

A **time-series momentum (TSMOM)** book across 77 global futures markets, following
Moskowitz-Ooi-Pedersen (2012) with Hurst-Ooi-Pedersen's (2017) book-level volatility
target: long markets up over the past 12 months, short those down, every position sized
inverse to its own volatility, the whole book held near 10% annualised risk.

Headline, net of 2bp per unit of notional traded:

| | Walk-forward OOS 2005-2014 | Held-out 2010-2014 |
|---|---|---|
| Sharpe | **1.08** (t = 3.49) | **1.27** (t = 2.90) |
| Bootstrap 95% CI | [0.42, 1.74] | [0.34, 2.27] |
| Max drawdown | −14.3% | −12.0% |

Two results the write-up leads on, in preference to a benchmark horse race:

- **As a portfolio sleeve.** A 20% allocation funded out of a 60/40 portfolio lifts its
  Sharpe 0.53 → 0.75 and cuts max drawdown −33.5% → −23.5% over the walk-forward decade.
- **Breadth is the mechanism.** Out-of-sample Sharpe rises monotonically with the number
  of markets traded (0.61 at five → 1.27 at 77); concentrating into the 20 most-correlated
  markets cuts it to 0.49.

The strategy's own Sharpe is statistically significant; its *margin* over a risk-matched
passive control is not (bootstrap interval on the difference straddles zero). The
write-up states this explicitly rather than leading on the point estimate.

## Layout

- `data/`: the raw provided CSVs (`data/raw/`, gitignored). See `data/README.md` for the
  expected layout.
- `src/cta/`: all real logic, imported (not copy-pasted) into notebooks:
  - `data.py`: load/clean/align, point-in-time universe screen
  - `indicators.py`: signal building blocks, incl. roll-safe futures returns
  - `strategies.py`: indicators → target positions
  - `backtest.py`: portfolio assembly, walk-forward, vol targeting, universe studies
  - `metrics.py`: Sharpe/max DD/turnover, plus block-bootstrap inference
  - `plotting.py`: shared chart styling (validated palette, fixed series colours)
- `notebooks/`
  - `01_data_exploration.ipynb`: EDA (scratch)
  - `02_strategy_research.ipynb`: working log (scratch): signal prototyping, the
    roll-gap bug and its fix, and Part 4's record of every variant tested and rejected
  - `03_final_writeup.ipynb`: **the deliverable**. Intro, methodology, findings,
    takeaways. Thin, mostly calls into `src/cta/`.
- `research/literature_notes.md`: paper summaries used as inspiration + benchmark
  definitions.
- `reports/figures/`: exported charts referenced from the final notebook.
- `tests/`: lightweight sanity checks on `src/cta/`.

## Setup

```bash
uv sync
```

The provided CSVs are not in the repository. Put them at `data/raw/Delta1/`, keeping the
`Futures Data/` and `ETF Data/` subdirectories and the two `CATALOGUE_*.csv` files as
supplied; `data/README.md` lists the expected layout. Nothing else is generated or cached,
so that is the only input required.

## Running

```bash
uv run jupyter lab          # open notebooks/
uv run pytest                # run tests
uv run ruff format .         # format
uv run ruff check .          # lint
```
