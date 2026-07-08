"""Phase 4 acceptance: the dispatcher respects physics and makes money on a known spread."""

import numpy as np
import pandas as pd
import pytest

from forecast_to_dispatch.config import load_config
from forecast_to_dispatch.optimize.dispatch import (
    ALL_PRODUCTS,
    UP_PRODUCTS,
    BatterySpec,
    expected_price_from_quantiles,
    lp_binary_check,
    optimize_dispatch,
)


@pytest.fixture(scope="module")
def spec() -> BatterySpec:
    return BatterySpec.from_config(load_config())


def _prices(energy: list[float], as_price: float = 0.0) -> pd.DataFrame:
    idx = pd.date_range("2024-05-15", periods=len(energy), freq="1h", tz="US/Central")
    df = pd.DataFrame({"energy_price": energy}, index=idx)
    for k in ALL_PRODUCTS:
        df[k] = as_price
    return df


CHEAP_NIGHT_DEAR_EVENING = [10.0] * 8 + [25.0] * 10 + [100.0] * 4 + [25.0] * 2


def test_positive_profit_on_known_spread(spec):
    schedule = optimize_dispatch(_prices(CHEAP_NIGHT_DEAR_EVENING), spec)
    assert schedule["expected_profit"].sum() > 0
    # Buy-low sell-high shape: charging concentrated in the $10 hours,
    # discharging in the $100 hours.
    cheap = schedule.iloc[:8]
    dear = schedule.iloc[18:22]
    assert cheap["charge_mw"].sum() > 0
    assert dear["discharge_mw"].sum() > 0
    assert dear["charge_mw"].sum() == pytest.approx(0, abs=1e-6)


def test_constraints_hold(spec):
    s = optimize_dispatch(_prices(CHEAP_NIGHT_DEAR_EVENING, as_price=5.0), spec)
    assert (s["soc_mwh"] >= spec.soc_min_mwh - 1e-6).all()
    assert (s["soc_mwh"] <= spec.soc_max_mwh + 1e-6).all()
    assert (s["charge_mw"] <= spec.power_mw + 1e-6).all()
    assert (s["discharge_mw"] <= spec.power_mw + 1e-6).all()
    up = sum(s[f"r_{k}_mw"] for k in UP_PRODUCTS)
    assert (s["discharge_mw"] + up <= spec.power_mw + 1e-3).all()
    assert (up / spec.eta_discharge <= s["soc_mwh"] - spec.soc_min_mwh + 1e-3).all()


def test_soc_dynamics_identity(spec):
    s = optimize_dispatch(_prices(CHEAP_NIGHT_DEAR_EVENING), spec)
    soc = np.concatenate([[spec.soc_initial_mwh], s["soc_mwh"].to_numpy()])
    step = spec.eta_charge * s["charge_mw"].to_numpy() - (
        s["discharge_mw"].to_numpy() / spec.eta_discharge
    )
    assert np.allclose(np.diff(soc), step, atol=1e-4)


def test_cyclic_terminal_soc(spec):
    s = optimize_dispatch(_prices(CHEAP_NIGHT_DEAR_EVENING), spec)
    assert s["soc_mwh"].iloc[-1] == pytest.approx(spec.soc_initial_mwh, abs=1e-3)


def test_no_simultaneous_charge_discharge_even_at_negative_prices(spec):
    # Negative prices are exactly where the LP relaxation wants to burn energy
    # both ways; the MILP escalation must forbid it.
    wild = [-40.0] * 6 + [5.0] * 6 + [-15.0] * 4 + [90.0] * 6 + [10.0] * 2
    s = optimize_dispatch(_prices(wild), spec)
    assert (s["charge_mw"] * s["discharge_mw"]).max() <= 1e-4


def test_reserves_pay_when_energy_is_flat(spec):
    # Flat energy prices, valuable reserves: the battery should sell standby
    # capacity instead of cycling.
    s = optimize_dispatch(_prices([30.0] * 24, as_price=12.0), spec)
    total_up = sum(s[f"r_{k}_mw"] for k in UP_PRODUCTS).sum()
    assert total_up > 0
    assert s["expected_profit"].sum() > 0
    # cycling should be minimal: degradation + zero spread makes it pointless
    assert s["charge_mw"].sum() < 10


def test_reserves_require_stored_energy(spec):
    # An empty battery must not promise up-reserves.
    empty = BatterySpec(
        power_mw=spec.power_mw,
        energy_mwh=spec.energy_mwh,
        eta_charge=spec.eta_charge,
        eta_discharge=spec.eta_discharge,
        soc_min_mwh=0.0,
        soc_max_mwh=spec.soc_max_mwh,
        soc_initial_mwh=0.0,
        degradation_cost_per_mwh=spec.degradation_cost_per_mwh,
    )
    # No spread to charge from, reserves priced high: without stored energy,
    # up-reserve awards must stay bounded by what it can charge up to.
    s = optimize_dispatch(_prices([1000.0] * 2 + [30.0] * 22, as_price=50.0), empty)
    up_first = sum(s[f"r_{k}_mw"].iloc[0] for k in UP_PRODUCTS)
    assert up_first / empty.eta_discharge <= s["soc_mwh"].iloc[0] + 1e-3


def test_lp_binary_check_reports(spec):
    wild = [-40.0] * 6 + [5.0] * 12 + [90.0] * 6
    report = lp_binary_check(_prices(wild), spec)
    assert set(report) >= {"lp_overlap_hours", "lp_expected_profit", "escalated"}
    if report["escalated"]:
        assert report["milp_expected_profit"] <= report["lp_expected_profit"] + 1e-6


def test_expected_price_from_quantiles_weights_tails():
    preds = pd.DataFrame(
        {"q05": [10.0], "q25": [20.0], "q50": [30.0], "q75": [40.0], "q95": [500.0]},
        index=pd.date_range("2024-05-15", periods=1, freq="1h", tz="US/Central"),
    )
    ep = expected_price_from_quantiles(preds, [0.05, 0.25, 0.50, 0.75, 0.95])
    assert ep.iloc[0] > 30.0, "a fat upper tail must pull the dispatch signal above the median"


def test_missing_price_columns_fail_loudly(spec):
    df = _prices(CHEAP_NIGHT_DEAR_EVENING).drop(columns=["ecrs"])
    with pytest.raises(ValueError, match="missing columns"):
        optimize_dispatch(df, spec)
