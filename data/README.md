# Data

`raw/` and `processed/` are gitignored (data was provided directly) - this file documents what should be there so the pipeline is reproducible for anyone with access to the source files.

## `raw/Delta1/`

As provided, untouched.

- `CATALOGUE_Delta1_ETF.csv` — metadata for every ETF symbol (name, exchange, base/sub
  type, business summary, first quoted date). Use this to find candidate govt-bond-ETF
  tickers (e.g. `subtype1`/`subtype2` filtering).
- `CATALOGUE_Delta1_Futures.csv` — metadata for every futures symbol (name, currency,
  exchange, `Class`, tick size, point value, margin). Continuous contracts are prefixed
  `&`; a `_CCB` suffix means back-adjusted continuous contract. G10 FX and Gold futures
  live here (e.g. `&6E` = Euro FX, `&6A` = AUD, ... ; gold is under `Class` metals).
- `ETF Data/<SYMBOL>.csv` — one file per ETF: `Date, Open, High, Low, Close, Volume,
  Turnover, Unadjusted Close, Dividend, Constituent_*` (S&P index membership flags).
- `Futures Data/<SYMBOL>.csv` — one file per futures contract: `Date, Open, High, Low,
  Close, Volume, Delivery Month, Open Interest`.

Universe selection (which specific FX/Gold/Bond symbols we use) happens in
`notebooks/01_data_exploration.ipynb`, driven off the two catalogue files rather than
hardcoded here.

## `processed/`

Cleaned, calendar-aligned, resampled output of `src/cta/data.py`, written here by the
pipeline — not committed, regenerate by re-running the notebook/pipeline.
