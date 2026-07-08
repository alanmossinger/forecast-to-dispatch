"""Phase 2 acceptance: the feature matrix cannot see the future.

Two layers of proof:
1. Structural — every feature in FEATURE_SPEC lags its source by at least the
   legal minimum implied by the issue-time convention (09:00 on D-1).
2. Mechanical — spot-check that lag features literally equal the source series
   shifted, and that truncating the input at the issue time reproduces the
   same feature rows for the delivery day (nothing after the issue time can
   change them).
"""

import numpy as np
import pandas as pd
import pytest

from forecast_to_dispatch.config import load_config
from forecast_to_dispatch.data.ingest import load_sample
from forecast_to_dispatch.features.build import (
    FEATURE_SPEC,
    MIN_LAG_BY_SOURCE,
    build_features,
    issue_time_for,
)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return load_sample(load_config())


@pytest.fixture(scope="module")
def features(panel):
    return build_features(panel)


def test_every_feature_respects_minimum_lag():
    for name, rule in FEATURE_SPEC.items():
        assert rule.lag_hours >= MIN_LAG_BY_SOURCE[rule.source], (
            f"{name} lags {rule.source} by only {rule.lag_hours}h — "
            f"below the {MIN_LAG_BY_SOURCE[rule.source]}h issue-time minimum"
        )


def test_feature_timestamps_precede_issue_time():
    """The spec's acceptance test, stated directly: for every feature, the most
    recent source observation it uses is timestamped at or before the moment
    the forecast for that delivery day was issued (except DAM/AS D-1 values,
    which clear on D-2 and are legitimately known)."""
    for name, rule in FEATURE_SPEC.items():
        if rule.source == "calendar":
            continue
        # Worst case: first hour of delivery day D (midnight) has the SMALLEST
        # gap between the lagged source timestamp and the issue time.
        target = pd.Timestamp("2024-05-15 23:00", tz="US/Central")  # last hour worst for dam
        newest_source_ts = target - pd.Timedelta(hours=rule.lag_hours)
        issue = issue_time_for(target)
        if rule.source in ("dam_price", "as_total"):
            # DAM values for delivery day D-1 were published on D-2 (~13:30),
            # so 'known' means: source delivery day <= D-1.
            assert newest_source_ts.normalize() <= issue.normalize(), name
        else:
            assert (
                newest_source_ts <= issue
            ), f"{name}: source ts {newest_source_ts} is after issue time {issue}"


def test_lag_features_are_mechanical_shifts(panel, features):
    X, _ = features
    t = X.index[len(X) // 2]
    assert X.loc[t, "rtm_lag48"] == panel.loc[t - pd.Timedelta(hours=48), "rtm_price"]
    assert X.loc[t, "dam_lag24"] == panel.loc[t - pd.Timedelta(hours=24), "dam_price"]
    assert X.loc[t, "rtm_lag168"] == panel.loc[t - pd.Timedelta(hours=168), "rtm_price"]


def test_rolling_features_end_before_issue_time(panel, features):
    X, _ = features
    t = X.index[len(X) // 2]
    window_end = t - pd.Timedelta(hours=48)
    expected = panel.loc[window_end - pd.Timedelta(hours=23) : window_end, "rtm_price"].mean()
    assert np.isclose(X.loc[t, "rtm_roll24_mean"], expected)


def test_truncation_at_issue_time_changes_nothing(panel):
    """Rebuild features using ONLY data available at the issue time of one
    delivery day; the rows for that day must be identical. This is the
    definitive no-leakage proof: nothing after the issue time can influence
    the features the model sees."""
    X_full, _ = build_features(panel)
    delivery_day = X_full.index[-24].normalize()
    issue = issue_time_for(delivery_day)

    blinded = panel.copy()
    # RTM prices (and the spread, which contains RTM) unknown from issue time on.
    blinded.loc[blinded.index >= issue, ["rtm_price", "dart_spread"]] = np.nan
    # DAM & AS for delivery day D clear at ~13:30 on D-1 — after the 09:00
    # issue — so they are unknown too; D-1 and earlier are known.
    blinded.loc[
        blinded.index >= delivery_day,
        ["dam_price", "regup", "regdn", "rrs", "nspin", "ecrs"],
    ] = np.nan

    X_blind, _ = build_features(blinded, require_targets=False)
    day_idx = X_full.index[X_full.index.normalize() == delivery_day]
    assert len(day_idx) == 24
    pd.testing.assert_frame_equal(X_full.loc[day_idx], X_blind.loc[day_idx])


def test_targets_align_with_panel(panel, features):
    X, targets = features
    assert list(targets.columns) == ["y_price", "y_spread"]
    t = X.index[100]
    assert targets.loc[t, "y_price"] == panel.loc[t, "rtm_price"]
    assert targets.loc[t, "y_spread"] == panel.loc[t, "dart_spread"]


def test_no_nans_and_reasonable_size(features):
    X, targets = features
    assert not X.isna().any().any()
    assert not targets.isna().any().any()
    assert len(X) == len(targets)
    assert len(X) >= 24 * 14, "at least two weeks of usable rows from the one-month sample"
