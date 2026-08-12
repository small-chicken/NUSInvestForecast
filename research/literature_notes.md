# Literature Notes

Paper summaries used as inspiration for the strategy, plus the benchmark definition(s)
we backtest against. One subsection per paper/source: citation, core idea, what we borrow,
what we deliberately simplify (Occam's razor).

## Momentum / Trend-following (CTA)

### Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum", *Journal of Financial
Economics* 104(2), 228-250. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2089463) · [PDF (NYU Stern)](https://w4.stern.nyu.edu/facdir/lpederse/papers/TimeSeriesMomentum.pdf)

The foundational paper for this strategy family - primary inspiration.

- **Universe**: 58 liquid futures - 24 commodities, 12 currency pairs, 12 equity indices,
  13 government bonds. Jan 1965-Dec 2009 (main tests from 1985, when breadth/liquidity
  are adequate).
- **Signal**: sign of the past 12-month excess return of each instrument. Long if
  positive, short if negative.
- **Position sizing**: inverse-volatility scaling. Ex-ante vol `σ_t` estimated from an
  exponentially-weighted average of squared daily returns, weights `(1-δ)δ^i`, `δ` chosen
  so the weighting scheme's center-of-mass is 60 days. Position size = `40% / σ_{t-1}`
  (the 40% is an arbitrary constant, chosen only so the resulting portfolio vol is
  comparable to other published factors - not load-bearing).
- **Return formula**: `r_TSMOM,t+1 = (1/S_t) Σ_s sign(r^s_{t-12,t}) · (40%/σ^s_t) · r^s_{t+1}`
- **Return construction (important)**: they build each instrument's excess return series
  by compounding daily returns of the most-liquid contract, rolling to the next contract
  as needed - i.e. returns are computed while holding a single contract and the return
  series is spliced across rolls, not the price level series. This sidesteps the
  back-adjustment sign issue we hit in our EDA entirely, because a % return is never taken
  across a roll-day price jump.
- **Benchmark**: a passive always-long position in the same instruments, same
  vol-scaling - diversified TSMOM clearly outperforms this (their Fig. 3: $100 → ~$100k
  for TSMOM vs. ~$5-10k for passive long, 1985-2009, log scale). Also benchmarked against
  MSCI World, Barclays Agg Bond, S&P GSCI, and Fama-French/Carhart factors (SMB, HML, UMD)
  - alpha survives all of them.
- **Headline results**: diversified portfolio Sharpe > 1 (roughly 2.5x the equity market's
  Sharpe over the same period), monthly alpha 1.09-1.58% (t-stats 5.4-8.0) depending on
  factor model, positive in every one of the 58 instruments (52/58 statistically
  significant). Performs best in extreme markets (a "smile" against S&P 500 returns) -
  crisis-alpha property.
- **What we borrow**: the sign-of-past-return signal, inverse-vol position sizing, and
  critically the return-construction convention (compound returns per held contract,
  don't take % returns across a difference-adjusted price level).
- **What we simplify**: single lookback/holding pair rather than their full 1-48 month
  grid (we replicate a slice of their robustness grid, not all of it); no CFTC
  speculator/hedger positioning analysis (out of scope, not needed to answer "does this
  strategy work").

### Hurst, Ooi & Pedersen (2017), "A Century of Evidence on Trend-Following Investing",
AQR working paper (published in *Journal of Portfolio Management*). [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2993026) · [PDF](https://static.twentyoverten.com/593e8a9e7299b471eaecf644/SkLoGL67M/A-Century-of-Evidence-on-Trend-Following-Investing.pdf)

Out-of-sample validation of the same methodology across 137 years - used for robustness
framing, not a different strategy.

- **Universe**: 67 markets - 29 commodities, 11 equity indices, 15 bonds, 12 currencies -
  1880-2016, including hand-transcribed 19th-century commodity data.
- **Signal**: equal-weighted blend of 1-month, 3-month, and 12-month time-series momentum
  (multiple horizons, not just 12-month) - a refinement over the single-lookback MOP
  strategy.
- **Position sizing**: per-market vol targeting as in MOP, plus portfolio-level vol
  targeting (10% annualized) using a rolling 3-year covariance matrix - an extra layer MOP
  doesn't do.
- **Headline results**: positive average return in **every decade since 1880** (Sharpe net
  of costs/fees ranges 0.13 to 1.70 by decade, full-sample net Sharpe 0.76). Correlation to
  US equities and bonds close to zero in every decade. Positive returns in 8 of the 10
  largest 60/40 portfolio drawdowns over the century.
- **Key robustness finding**: performance is largely indifferent to growth/inflation/war
  regimes - the one variable that actually predicts trend-following performance is
  **average cross-market correlation** (low correlation → better performance; high
  correlation, e.g. 2008-2014 "risk-on/risk-off" era → worse). Directly relevant to our
  own correlation checks from the EDA notebook.
- **What we borrow**: the multi-horizon blend idea (as an optional enhancement, not
  required for a baseline) and the framing that correlation regime, not macro regime, is
  the thing to watch.
- **What we simplify**: we don't attempt 137 years of history (our data caps at 2014
  anyway) or hand-collect pre-1978 data.

### Baltas & Kosowski (2013), "Momentum Strategies in Futures Markets and Trend-Following
Funds", Imperial College Business School working paper. [SSRN](https://ssrn.com/abstract=1968996) · [PDF](https://www.naaim.org/wp-content/uploads/2013/10/00S_Momentum_Strategies_in_Futures_Markets_Nick_Baltas.pdf)

The closest match to our own setup - universe and sample period nearly identical to ours.

- **Universe**: 71 futures - 26 commodities, 23 equity indices, 7 currencies, 15 bonds -
  **Dec 1974-Jan 2012**, near-identical to our 1978-2014 dataset both in breadth and era.
- **Return construction**: identical philosophy to MOP - splice the most-liquid contract
  at each point in time (rolling by daily tick volume), forward-fill for exchange holiday
  misalignment, then take daily excess close-to-close % returns on that spliced series.
  Confirms our EDA's back-adjustment concern is a data-provider-specific issue, not an
  inherent property of continuous futures - the standard academic convention avoids it by
  construction rather than by using difference-adjusted series.
- **Position sizing**: same `sign(J-month return) × 40%/σ_i(t;60) × R_i` formula as MOP,
  with the same 40% scaling constant (validated as still reasonable for their sample: their
  strategies realize 12.6-15.3% annualized vol, in the same range as MSCI/UMD factor vol).
- **Volatility estimator refinement**: notes MOP's simple exponentially-weighted
  squared-return estimator can be replaced by the **Yang & Zhang (2000) range estimator**,
  which uses open/high/low/close (not just close-to-close) and is more statistically
  efficient - directly usable for us since our futures data includes full OHLC, not just
  Close. Worth trying as a documented "optimization" in the methodology write-up, not
  required for a baseline.
- **Multi-frequency grid**: tests monthly (J,K ∈ {1,3,6,9,12,24,36}), weekly
  (∈ {1,2,3,4,6,8,12}), and daily (∈ {1,3,5,10,15,30,60}) lookback/holding combinations -
  Sharpe ratios above 1.20 achievable at multiple frequencies, and momentum strategies at
  different frequencies have low cross-correlation (they capture distinct patterns, not
  redundant ones).
- **CTA linkage**: regresses an AUM-weighted index of systematic CTA funds (BarclayHedge,
  1348 funds) on these time-series momentum strategies - coefficients are highly
  significant even controlling for standard factors, i.e. real-world CTA funds' returns
  are substantially explained by exactly this kind of signal. Supporting evidence that this
  strategy family isn't just an academic curiosity.
- **What we borrow**: the return-construction convention (resolves our EDA's open
  methodological question), the 40% vol-scaling constant as a starting point, and the
  lookback/holding grid structure for our own robustness/sensitivity check.
- **What we simplify**: no capacity-constraint analysis (irrelevant at our scale), no CTA
  fund-index regression (would need a licensed data source we don't have) - we note the
  finding as supporting context rather than reproducing it.

## Regime Allocation

Out of scope for the direction chosen (momentum/trend-following CTA across the broad
futures universe, not ETF regime rotation) - left here for context on why it wasn't
pursued, not as a live research thread.

## Volatility Strategies

Not pursued as a standalone strategy; volatility shows up throughout as the *position
sizing* mechanism (inverse-vol weighting, per MOP/Hurst/Baltas above) rather than as the
trading signal itself. If we revisit this, Baltas & Kosowski's Yang-Zhang range estimator
note above is the natural entry point.

## Benchmark Definitions

**Primary benchmark: passive long, equal-vol-weighted buy-and-hold** of the same curated
84-instrument universe from `01_data_exploration.ipynb` - always long every instrument,
sized to the same per-instrument vol target as the trend strategy, no signal/timing at
all. This is exactly the comparison MOP use in their Fig. 3 ("does timing add value over
just holding the same risk exposure") and is trivially explainable - the cleanest test of
whether the trend signal itself is doing anything.

**Secondary, for robustness**: a lookback/holding sensitivity grid (12-month lookback /
1-month holding as the primary choice, following MOP/Baltas's strongest and most-cited
result; a handful of neighbours from Baltas's grid as a check that the choice isn't
cherry-picked) rather than a single untested parameter pair.

**Return construction, carried forward from the EDA notebook's open question**: use the
**raw (non-`_CCB`) continuous contract series**, daily % returns, not the back-adjusted
series - this matches MOP/Baltas's own convention (splice the most-liquid contract, take
% returns on that) and avoids the non-positive-price issue found in `01_data_exploration.ipynb`
without needing an ad hoc price-difference workaround. Roll-day noise (the one day the
underlying contract switches) is a known, accepted simplification in this literature
rather than something to eliminate - Baltas & Kosowski use the identical convention over a
near-identical universe and era.
