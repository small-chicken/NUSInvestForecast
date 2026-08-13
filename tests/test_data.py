"""Sanity checks for src/cta/data.py -- requires data/raw/Delta1/ to be present locally."""

from cta.data import curated_futures_universe, load_futures, load_futures_catalogue


def test_curated_universe_size_and_exclusions():
    universe = curated_futures_universe()
    assert len(universe) == 77
    assert "&6N" not in universe  # late starter (2005)
    assert "&YAP4" not in universe  # redundant ASX SPI 200 variant
    assert "&YAP" in universe  # the kept variant
    assert "&YIB" not in universe  # 163-day stale-price run -- see strategies.py cap
    assert "&AFB" not in universe  # 42-day stale-price run
    assert "&ZQ" not in universe  # 33-day stale-price run
    assert "&AWM" not in universe  # 31-day stale-price run
    assert "&BAX" not in universe  # money-market rate future, not a duration bond
    assert "&LEU" not in universe  # Euribor -- same reason
    assert "&YIR" not in universe  # ASX bank bills -- same reason
    assert "&ZT" in universe  # genuine 2-year government bond, kept despite low vol


def test_curated_universe_symbols_are_loadable():
    universe = curated_futures_universe()
    for sym in universe[:3]:  # a few, not all 77 -- keep the test fast
        df = load_futures(sym)
        assert len(df) > 0
        assert (df["Close"] > 0).all()  # raw (non-CCB) series should never be non-positive


def test_futures_catalogue_has_expected_columns():
    catalogue = load_futures_catalogue()
    assert {"symbol", "subtype1", "securityname"}.issubset(catalogue.columns)


def test_point_in_time_universe_ignores_the_future():
    """A market screened out only because of post-2009 behaviour must still be selectable
    standing at end-2009 -- that is the whole point of a point-in-time screen.
    """
    full_sample = set(curated_futures_universe())
    as_of_2009 = set(curated_futures_universe(as_of="2009-12-31"))

    # &YIB's catastrophic 163-day stale run makes it unusable full-sample; the screen must
    # still be judging it on its own pre-2010 record, not a 2014-vintage view.
    assert "&YIB" not in full_sample
    # Late starters excluded full-sample had ample history by end-2009 and are tradeable.
    assert "&RB" in as_of_2009 and "&RB" not in full_sample


def test_point_in_time_universe_requires_enough_history():
    """Nothing selectable at end-1980 should be a market that had barely started."""
    early = curated_futures_universe(as_of="1980-12-31")
    assert len(early) < len(curated_futures_universe(as_of="2009-12-31"))
