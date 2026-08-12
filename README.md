# NUS Investment Society QR Recruitment — Q3

Trading strategy (momentum/trend-following CTA, dynamic regime allocation, or vol
strategy) on G10 FX / Gold Continuous Futures / Govt Bond ETFs or Futures, with ≥5 years
of out-of-sample walk-forward results, benchmarked against literature.

## Layout

- `data/` — raw provided CSVs (`data/raw/`, gitignored) and cleaned/aligned output
  (`data/processed/`, gitignored). See `data/README.md`.
- `src/cta/` — all real logic, imported (not copy-pasted) into notebooks:
  - `data.py` — load/clean/align/resample
  - `indicators.py` — signal building blocks
  - `strategies.py` — indicators → target positions
  - `backtest.py` — vectorbt wrapper, incl. walk-forward
  - `metrics.py` — Sharpe/Sortino/max DD/turnover/benchmark comparison
  - `plotting.py` — shared chart styling
- `notebooks/`
  - `01_data_exploration.ipynb` — EDA (scratch)
  - `02_strategy_research.ipynb` — signal prototyping (scratch)
  - `03_final_writeup.ipynb` — **the deliverable**: intro, methodology, findings,
    takeaways. Thin — mostly calls into `src/cta/`.
- `research/literature_notes.md` — paper summaries used as inspiration + benchmark
  definitions.
- `reports/figures/` — exported charts referenced from the final notebook.
- `tests/` — lightweight sanity checks on `src/cta/`.

## Setup

```bash
uv sync
```

## Running

```bash
uv run jupyter lab          # open notebooks/
uv run pytest                # run tests
uv run ruff format .         # format
uv run ruff check .          # lint
```
