# Data

`raw/` is gitignored, since the data was provided directly and redistribution is not
confirmed. This file documents what should be there, so the pipeline is reproducible for
anyone with access to the source files.

## `raw/Delta1/`

As provided, untouched.

- `CATALOGUE_Delta1_ETF.csv`: metadata for every ETF symbol (name, exchange, base and sub
  type, business summary, first quoted date). Used to find candidate government-bond ETF
  tickers, by filtering `subtype1` and `subtype2`.
- `CATALOGUE_Delta1_Futures.csv`: metadata for every futures symbol (name, currency,
  exchange, `Class`, tick size, point value, margin). Continuous contracts are prefixed `&`
  and a `_CCB` suffix marks the back-adjusted continuous contract, so `&6E` is the raw Euro
  FX continuous series and `&6E_CCB` its back-adjusted twin.
- `ETF Data/<SYMBOL>.csv`: one file per ETF: `Date, Open, High, Low, Close, Volume,
  Turnover, Unadjusted Close, Dividend, Constituent_*` (S&P index membership flags).
- `Futures Data/<SYMBOL>.csv`: one file per futures contract: `Date, Open, High, Low,
  Close, Volume, Delivery Month, Open Interest`.

`src/cta/data.py` resolves this location as `<repo root>/data/raw/Delta1`, so the CSVs have
to sit at exactly that path for the notebooks to run.

Universe selection, meaning which futures symbols the strategy trades, happens in
`cta.data.curated_futures_universe()`, driven off the two catalogue files rather than
hardcoded here.

## There is no intermediate stage

Nothing is cached or pre-processed to disk. `src/cta/data.py` loads, cleans and aligns from
`raw/` in memory on every run and writes no files, so `raw/` is the only input a reviewer
needs to supply. A full run of `03_final_writeup.ipynb` takes about 30 seconds, which is
why no caching layer was added.
