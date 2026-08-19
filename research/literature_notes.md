# Literature Notes

Paper summaries used as sources for the strategy, and the benchmark definitions the
backtest runs against. One subsection per source: citation, core idea, what is adopted
here, and what is simplified.

## Momentum / trend-following (CTA)

### Moskowitz, Ooi & Pedersen (2012)

"Time Series Momentum", *Journal of Financial Economics* 104(2), 228-250.
[SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2089463) ·
[PDF (NYU Stern)](https://w4.stern.nyu.edu/facdir/lpederse/papers/TimeSeriesMomentum.pdf)

The foundational paper for this strategy family and the primary source for the
specification used here.

- **Universe**: 58 liquid futures: 24 commodities, 12 currency pairs, 12 equity indices,
  13 government bonds. January 1965 to December 2009, with main tests from 1985 onward,
  when breadth and liquidity are adequate.
- **Signal**: the sign of the past 12-month excess return of each instrument. Long if
  positive, short if negative.
- **Position sizing**: inverse-volatility scaling. Ex-ante vol `σ_t` is estimated from an
  exponentially weighted average of squared daily returns, weights `(1-δ)δ^i`, with `δ`
  chosen so the weighting scheme's centre of mass is 60 days. Position size is
  `40% / σ_{t-1}`. The 40% is an arbitrary constant, chosen so the resulting portfolio vol
  is comparable to other published factors, and is not load-bearing.
- **Return formula**: `r_TSMOM,t+1 = (1/S_t) Σ_s sign(r^s_{t-12,t}) · (40%/σ^s_t) · r^s_{t+1}`
- **Return construction**: each instrument's excess return series is built by compounding
  daily returns of the most-liquid contract and rolling to the next contract as needed.
  Returns are computed while holding a single contract and the return series is spliced
  across rolls, rather than the price level series. This avoids the back-adjustment sign
  issue encountered in the EDA, because a percentage return is never taken across a
  roll-day price jump.
- **Benchmark**: a passive always-long position in the same instruments under the same
  vol-scaling. Diversified TSMOM outperforms it (Fig. 3: $100 to ~$100k for TSMOM against
  ~$5-10k for passive long, 1985-2009, log scale). Also benchmarked against MSCI World,
  Barclays Agg Bond, S&P GSCI, and the Fama-French/Carhart factors (SMB, HML, UMD); alpha
  survives all of them.
- **Headline results**: diversified portfolio Sharpe above 1, roughly 2.5× the equity
  market's Sharpe over the same period. Monthly alpha 1.09-1.58% (t-stats 5.4-8.0)
  depending on factor model. Returns are positive in all 58 instruments, statistically
  significant in 52. Performance is strongest in extreme markets, producing a smile
  against S&P 500 returns, the crisis-alpha property.
- **Adopted here**: the sign-of-past-return signal, inverse-vol position sizing, and the
  return-construction convention (compound returns per held contract rather than
  percentage returns across a difference-adjusted price level).
- **Simplified**: a single lookback/holding pair rather than the full 1-48 month grid,
  replicating a slice of the robustness grid rather than all of it. The CFTC
  speculator/hedger positioning analysis is out of scope.

### Hurst, Ooi & Pedersen (2017)

"A Century of Evidence on Trend-Following Investing", AQR working paper, published in
*Journal of Portfolio Management*.
[SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2993026) ·
[PDF](https://static.twentyoverten.com/593e8a9e7299b471eaecf644/SkLoGL67M/A-Century-of-Evidence-on-Trend-Following-Investing.pdf)

Out-of-sample validation of the same methodology across 137 years. Used for robustness
framing rather than as a different strategy.

- **Universe**: 67 markets: 29 commodities, 11 equity indices, 15 bonds, 12 currencies.
  1880-2016, including hand-transcribed 19th-century commodity data.
- **Signal**: an equal-weighted blend of 1-month, 3-month and 12-month time-series
  momentum, a refinement over the single-lookback MOP strategy.
- **Position sizing**: per-market vol targeting as in MOP, plus portfolio-level vol
  targeting at 10% annualised using a rolling 3-year covariance matrix, a layer MOP does
  not have.
- **Headline results**: a positive average return in every decade since 1880, with Sharpe
  net of costs and fees ranging from 0.13 to 1.70 by decade and a full-sample net Sharpe
  of 0.76. Correlation to US equities and bonds is close to zero in every decade. Returns
  are positive in 8 of the 10 largest 60/40 portfolio drawdowns over the century.
- **Robustness finding**: performance is largely indifferent to growth, inflation and war
  regimes. The one variable that predicts trend-following performance is average
  cross-market correlation: low correlation gives better performance, high correlation
  (the 2008-2014 risk-on/risk-off era) worse.
- **Adopted here**: the portfolio-level vol target, which becomes the book-level risk dial
  in the final specification, and the framing that correlation regime rather than macro
  regime is the variable to watch. The multi-horizon blend was tested and not adopted; see
  `notebooks/02_strategy_research.ipynb` §4.2.
- **Simplified**: no attempt at 137 years of history, since the dataset ends in 2014, and
  no hand-collection of pre-1978 data.

### Baltas & Kosowski (2013)

"Momentum Strategies in Futures Markets and Trend-Following Funds", Imperial College
Business School working paper. [SSRN](https://ssrn.com/abstract=1968996) ·
[PDF](https://www.naaim.org/wp-content/uploads/2013/10/00S_Momentum_Strategies_in_Futures_Markets_Nick_Baltas.pdf)

The closest match to the setup used here in both universe and sample period.

- **Universe**: 71 futures: 26 commodities, 23 equity indices, 7 currencies, 15 bonds.
  December 1974 to January 2012, close to this project's 1978-2014 dataset in both breadth
  and era.
- **Return construction**: the same approach as MOP. Splice the most-liquid contract at
  each point in time, rolling by daily tick volume, forward-fill for exchange holiday
  misalignment, then take daily excess close-to-close percentage returns on the spliced
  series. This confirms that the EDA's back-adjustment problem is specific to the data
  provider rather than an inherent property of continuous futures: the standard academic
  convention avoids it by construction rather than by using difference-adjusted series.
- **Position sizing**: the same `sign(J-month return) × 40%/σ_i(t;60) × R_i` formula as
  MOP, with the same 40% scaling constant. Their strategies realise 12.6-15.3% annualised
  vol, in the same range as MSCI and UMD factor vol.
- **Volatility estimator refinement**: MOP's exponentially weighted squared-return
  estimator can be replaced by the Yang & Zhang (2000) range estimator, which uses
  open/high/low/close rather than close-to-close and is more statistically efficient. It is
  usable here, since the futures data includes full OHLC, but is not required for a
  baseline and was not implemented.
- **Multi-frequency grid**: tests monthly (J,K ∈ {1,3,6,9,12,24,36}), weekly
  (∈ {1,2,3,4,6,8,12}) and daily (∈ {1,3,5,10,15,30,60}) lookback/holding combinations.
  Sharpe ratios above 1.20 are achievable at multiple frequencies, and momentum strategies
  at different frequencies have low cross-correlation, so they capture distinct patterns
  rather than redundant ones.
- **CTA linkage**: an AUM-weighted index of systematic CTA funds (BarclayHedge, 1348 funds)
  is regressed on these time-series momentum strategies. Coefficients are significant even
  controlling for standard factors, so real-world CTA returns are substantially explained
  by this class of signal.
- **Adopted here**: the return-construction convention, which resolves the EDA's open
  methodological question, the 40% vol-scaling constant, and the lookback/holding grid
  structure used for the sensitivity check.
- **Simplified**: no capacity-constraint analysis, which is irrelevant at this scale, and
  no CTA fund-index regression, which would require a licensed data source. The finding is
  noted as supporting context rather than reproduced.

## Regime allocation

Out of scope for the direction chosen, which is momentum/trend-following CTA across the
broad futures universe rather than ETF regime rotation. Recorded here as context on why it
was not pursued rather than as a live research thread.

## Volatility strategies

Not pursued as a standalone strategy. Volatility appears throughout as the position-sizing
mechanism (inverse-vol weighting, per the three papers above) rather than as the trading
signal. Baltas & Kosowski's Yang-Zhang range estimator note above is the entry point if
this is revisited.

## Return construction

The convention originally recorded in this file was wrong, and the error inverted the
project's headline result.

The original note specified the raw (non-`_CCB`) continuous contract series with daily
percentage returns, and stated that roll-day noise is an accepted simplification that
Baltas & Kosowski use. Both claims were false, and both contradicted what this file records
about MOP and Baltas-Kosowski above: those papers splice the return series precisely so
that a percentage return is never taken across a roll-day price jump.

A raw continuous contract splices consecutive delivery months, so on each roll date the
price jumps by the calendar spread, and `pct_change()` books that jump as tradeable P&L.
It is not noise. It is systematically signed by contango and backwardation, so an
always-long benchmark harvests it while a long/short trend book pays it: roughly 25% of the
passive benchmark's entire 2005-2014 P&L came from the 2.3% of days that are rolls. The
construction implied that a rolling long VIX futures position returned +30.3% over
2005-2014, against a true -99.9%, and long WTI +26.5% against a true -58.3%.

The convention used is `held_contract_returns` (`src/cta/indicators.py`): the back-adjusted
price difference divided by the raw previous close. This is MOP's held-contract convention.
It also resolves the dilemma that produced the original error, since `_CCB` levels go
non-positive for 14 of 94 markets but that affects levels rather than differences, and the
raw close used as the denominator is never non-positive.

Correcting this reversed the headline: diversified TSMOM moves from Sharpe 0.64 to 1.05
over 2005-2014 while the passive benchmark falls from 1.06 to 0.80, and the corrected
figure agrees with MOP's reported Sharpe above 1. `notebooks/02_strategy_research.ipynb`
reproduces the contaminated result alongside the fix via `naive_spliced_returns`.

## Benchmark definitions

**Primary benchmark: passive long, equal-vol-weighted buy-and-hold** of the curated
universe from `01_data_exploration.ipynb`, later reduced to 77 instruments in
`cta.data.curated_futures_universe()`. Always long every instrument, sized to the same
per-instrument vol target as the trend strategy, with no timing signal. This is the
comparison MOP use in their Fig. 3, testing whether timing adds value over holding the same
risk exposure, and it is the cleanest test of whether the trend signal itself does
anything.

**Secondary, for robustness**: a lookback/holding sensitivity grid. A 12-month lookback
with 1-month holding is the primary choice, following MOP and Baltas-Kosowski's strongest
and most-cited result, with a set of neighbours from Baltas's grid as a check that the
choice is not cherry-picked.
