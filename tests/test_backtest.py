"""Phase 5 acceptance: settlement is exact, and the honesty invariants hold."""

import numpy as np
import pandas as pd
import pytest

from forecast_to_dispatch.config import load_config, resolve_path
from forecast_to_dispatch.optimize.dispatch import ALL_PRODUCTS, BatterySpec
from forecast_to_dispatch.backtest.engine import (
    climatology_price_curve,
    settle_on_realized_prices,
)


@pytest.fixture(scope="module")
def spec() -> BatterySpec:
    return BatterySpec.from_config(load_config())


def test_settlement_arithmetic_by_hand(spec):
    """One hour, hand-computed: settlement must be exact, not approximate."""
    idx = pd.date_range("2024-05-15", periods=1, freq="1h", tz="US/Central")
    schedule = pd.DataFrame(
        {
            "charge_mw": [0.0],
            "discharge_mw": [50.0],
            "r_regup_mw": [10.0],
            "r_regdn_mw": [0.0],
            "r_rrs_mw": [0.0],
            "r_nspin_mw": [20.0],
            "r_ecrs_mw": [0.0],
        },
        index=idx,
    )
    realized = pd.DataFrame(
        {
            "rtm_price": [200.0],
            "regup": [15.0],
            "regdn": [3.0],
            "rrs": [4.0],
            "nspin": [8.0],
            "ecrs": [9.0],
        },
        index=idx,
    )
    settled = settle_on_realized_prices(schedule, realized, spec)
    assert settled["energy"].iloc[0] == 200.0 * 50.0
    assert settled[["regup", "nspin"]].iloc[0].tolist() == [150.0, 160.0]
    expected_net = 10000 + 150 + 160 - spec.degradation_cost_per_mwh * 50
    assert settled["net_revenue"].iloc[0] == pytest.approx(expected_net)


def test_settlement_charges_cost_money(spec):
    idx = pd.date_range("2024-05-15", periods=1, freq="1h", tz="US/Central")
    schedule = pd.DataFrame(
        {"charge_mw": [100.0], "discharge_mw": [0.0]} | {f"r_{k}_mw": [0.0] for k in ALL_PRODUCTS},
        index=idx,
    )
    realized = pd.DataFrame({"rtm_price": [30.0]} | {k: [0.0] for k in ALL_PRODUCTS}, index=idx)
    settled = settle_on_realized_prices(schedule, realized, spec)
    assert settled["net_revenue"].iloc[0] == pytest.approx(-30.0 * 100 - 2.0 * 100)


def test_climatology_curve_shape():
    config = load_config()
    from forecast_to_dispatch.data.ingest import load_sample

    panel = load_sample(config)
    curve = climatology_price_curve(panel)
    assert len(curve) == 24
    # The evening hours must be pricier than pre-dawn — the naive policy's
    # whole reason to exist.
    assert curve.loc[19:21].mean() > curve.loc[2:4].mean()


FULL_RESULTS_KEYS = {"n_days", "totals", "revenue_capture_pct", "uplift_vs_naive_pct"}


@pytest.fixture(scope="module")
def full_daily():
    processed = resolve_path(load_config(), "processed")
    f = processed / "backtest_daily.parquet"
    if not f.exists():
        pytest.skip("full backtest not run yet (needs full study data)")
    daily = pd.read_parquet(f)
    if len(daily) < 30:
        pytest.skip("sample-sized backtest — invariants checked on the full run")
    return daily


def test_perfect_is_a_true_ceiling_every_single_day(full_daily):
    """Per-day invariant: no feasible policy can settle above the optimizer
    run on realized prices. A violation means settlement or dispatch is buggy."""
    assert (full_daily["perfect_revenue"] >= full_daily["governed_revenue"] - 1e-6).all()
    assert (full_daily["perfect_revenue"] >= full_daily["naive_revenue"] - 1e-6).all()


def test_governed_beats_naive_overall(full_daily):
    assert full_daily["governed_revenue"].sum() > full_daily["naive_revenue"].sum()


def test_capture_is_a_sane_fraction(full_daily):
    capture = full_daily["governed_revenue"].sum() / full_daily["perfect_revenue"].sum()
    assert 0.2 < capture < 1.0, (
        f"capture {capture:.2f} out of plausible range — near 1.0 suggests leakage, "
        "near 0 suggests a broken dispatch signal"
    )


def test_gap_correlates_with_forecast_error(full_daily):
    gap = full_daily["perfect_revenue"] - full_daily["governed_revenue"]
    rho = np.corrcoef(full_daily["forecast_mae"], gap)[0, 1]
    assert rho > 0, "worse forecasts should cost more money; a negative link means a bug"
