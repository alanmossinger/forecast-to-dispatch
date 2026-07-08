"""Phase 1 acceptance: the committed sample is a valid, aligned, offline price panel."""

import pandas as pd
import pytest

from forecast_to_dispatch.config import load_config
from forecast_to_dispatch.data.ingest import PANEL_COLUMNS, load_sample


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    config = load_config()
    try:
        return load_sample(config)
    except FileNotFoundError:
        pytest.skip("Committed sample not yet created (run live ingest with --make-sample)")


def test_sample_has_canonical_schema(sample):
    assert list(sample.columns) == PANEL_COLUMNS
    assert sample.index.name == "interval_start"
    assert sample.index.tz is not None, "index must be timezone-aware"


def test_sample_is_hourly_and_gapless(sample):
    """A hole in the price series would silently corrupt every revenue number."""
    expected = pd.date_range(sample.index.min(), sample.index.max(), freq="1h")
    assert len(sample) == len(expected)
    assert sample.index.equals(expected)
    assert not sample[PANEL_COLUMNS].isna().any().any()


def test_spread_identity(sample):
    """dart_spread must equal dam - rtm exactly — it is derived, never independent."""
    recomputed = sample["dam_price"] - sample["rtm_price"]
    pd.testing.assert_series_equal(
        sample["dart_spread"], recomputed, check_names=False, atol=1e-9, rtol=0
    )


def test_prices_are_plausible(sample):
    """ERCOT's offer cap bounds prices; violations mean a parsing bug, not a market event."""
    assert sample["rtm_price"].between(-300, 5100).all()
    assert sample["dam_price"].between(-300, 5100).all()
    # AS clearing prices are non-negative by market rule
    for col in ["regup", "regdn", "rrs", "nspin", "ecrs"]:
        assert (sample[col] >= 0).all(), f"{col} has negative clearing prices"


def test_sample_covers_config_window(sample):
    config = load_config()
    lo = pd.Timestamp(config["dates"]["sample_start"], tz=str(sample.index.tz))
    assert sample.index.min() == lo
    assert len(sample) >= 24 * 21, "sample must be at least three weeks for a meaningful demo"
