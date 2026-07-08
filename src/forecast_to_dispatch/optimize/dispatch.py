"""Battery dispatch optimizer: co-optimized energy + ancillary services (cvxpy).

Why this matters: this module is the *decision-maker* — it converts a price
forecast into the hourly schedule an operator would actually bid: when to buy
energy, when to sell it, and when to sell standby capacity (reserves) instead.
Under ERCOT's RTC+B design, energy and ancillary services clear together, and
a battery that only arbitrages energy leaves a large fraction of its value on
the table; the reserve headroom constraints below are what make the stacking
physically honest (a MW promised as Reg-Up must be backed by real discharge
headroom AND real stored energy).

The model (price-taker LP, hourly steps over one delivery day):
    max  Σ_t [ p_t (d_t - c_t) + Σ_k a_kt r_kt - c_deg (c_t + d_t) ]
    s.t. soc_{t+1} = soc_t + η_c c_t - d_t/η_d          (storage physics)
         soc_min ≤ soc ≤ soc_max, soc_0 = soc_T = initial (cyclic day)
         c_t, d_t ∈ [0, P_max]
         d_t + Σ_up r_kt ≤ P_max                         (up-reserve power headroom)
         c_t + r_dn,t   ≤ P_max                          (down-reserve power headroom)
         Σ_up r_kt / η_d ≤ soc_{t+1} - soc_min           (energy behind up reserves)
         η_c r_dn,t ≤ soc_max - soc_{t+1}                (room behind down reserves)

Simultaneous charge+discharge: with positive prices and a degradation cost the
LP never does it, but ERCOT prices go NEGATIVE, where burning energy through
the round-trip loss is 'profitable'. We solve the LP first and, only if a
violation appears, escalate to a MILP (HiGHS) with an exclusivity binary —
measured escalation instead of paying MILP cost every day.

Modeling honesty (stated in the model card):
- Price-taker: 100 MW at one hub does not move ERCOT prices.
- Reserves are paid on capacity ($/MW·h); deployment energy is not modeled.
- The optimizer sees FORECAST prices only; settlement on realized prices
  happens in the backtest (Phase 5). This file never touches realized prices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd

UP_PRODUCTS = ["regup", "rrs", "nspin", "ecrs"]
DOWN_PRODUCTS = ["regdn"]
ALL_PRODUCTS = UP_PRODUCTS + DOWN_PRODUCTS

SIMULTANEOUS_TOL_MW = 1e-4


@dataclass(frozen=True)
class BatterySpec:
    """Physical asset parameters — always constructed from config, never inline."""

    power_mw: float
    energy_mwh: float
    eta_charge: float
    eta_discharge: float
    soc_min_mwh: float
    soc_max_mwh: float
    soc_initial_mwh: float
    degradation_cost_per_mwh: float

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BatterySpec":
        b = config["battery"]
        return cls(
            power_mw=b["power_mw"],
            energy_mwh=b["energy_mwh"],
            eta_charge=b["eta_charge"],
            eta_discharge=b["eta_discharge"],
            soc_min_mwh=b["soc_min_mwh"],
            soc_max_mwh=b["soc_max_mwh"],
            soc_initial_mwh=b["soc_initial_mwh"],
            degradation_cost_per_mwh=b["degradation_cost_per_mwh"],
        )


def optimize_dispatch(
    prices: pd.DataFrame,
    spec: BatterySpec,
    allow_binaries: bool = True,
) -> pd.DataFrame:
    """Solve one horizon (typically 24 hours) of co-optimized dispatch.

    ``prices`` columns: ``energy_price`` ($/MWh, the price signal the operator
    believes — a forecast in production, realized only for the perfect-foresight
    bound) and one column per ancillary product (``regup``..``ecrs``, $/MW·h).

    Returns an hourly schedule with charge/discharge/soc, reserve awards, and
    an *expected* revenue decomposition at the given prices. Raises if the
    solver fails or any physical constraint is violated post-solve.
    """
    required = ["energy_price"] + ALL_PRODUCTS
    missing = [c for c in required if c not in prices.columns]
    if missing:
        raise ValueError(f"prices frame missing columns: {missing}")

    schedule = _solve(prices, spec, binaries=False)
    overlap = (schedule["charge_mw"] * schedule["discharge_mw"]).max()
    if overlap > SIMULTANEOUS_TOL_MW and allow_binaries:
        # LP relaxation chose to charge and discharge at once (negative-price
        # hours make wasting energy 'profitable') — escalate to exact MILP.
        schedule = _solve(prices, spec, binaries=True)

    _assert_physical(schedule, spec)
    return schedule


def _solve(prices: pd.DataFrame, spec: BatterySpec, binaries: bool) -> pd.DataFrame:
    T = len(prices)
    p = prices["energy_price"].to_numpy()

    charge = cp.Variable(T, nonneg=True)
    discharge = cp.Variable(T, nonneg=True)
    soc = cp.Variable(T + 1)
    reserves = {k: cp.Variable(T, nonneg=True) for k in ALL_PRODUCTS}

    cons = [
        soc[0] == spec.soc_initial_mwh,
        soc[T] == spec.soc_initial_mwh,  # cyclic day: no free energy at midnight
        soc[1:] == soc[:-1] + spec.eta_charge * charge - discharge / spec.eta_discharge,
        soc >= spec.soc_min_mwh,
        soc <= spec.soc_max_mwh,
        charge <= spec.power_mw,
        discharge <= spec.power_mw,
    ]
    up_total = cp.sum([reserves[k] for k in UP_PRODUCTS])
    dn_total = cp.sum([reserves[k] for k in DOWN_PRODUCTS])
    cons += [
        discharge + up_total <= spec.power_mw,
        charge + dn_total <= spec.power_mw,
        # Reserves must be energy-backed for a one-hour deployment.
        up_total / spec.eta_discharge <= soc[1:] - spec.soc_min_mwh,
        spec.eta_charge * dn_total <= spec.soc_max_mwh - soc[1:],
    ]
    if binaries:
        z = cp.Variable(T, boolean=True)  # 1 = charging mode
        cons += [charge <= spec.power_mw * z, discharge <= spec.power_mw * (1 - z)]

    as_revenue = cp.sum(
        cp.sum([cp.multiply(prices[k].to_numpy(), reserves[k]) for k in ALL_PRODUCTS])
    )
    energy_revenue = p @ (discharge - charge)
    degradation = spec.degradation_cost_per_mwh * cp.sum(charge + discharge)

    problem = cp.Problem(cp.Maximize(energy_revenue + as_revenue - degradation), cons)
    problem.solve(solver=cp.HIGHS)
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Dispatch optimization failed: status={problem.status}")

    out = pd.DataFrame(index=prices.index)
    out["charge_mw"] = np.clip(charge.value, 0, None)
    out["discharge_mw"] = np.clip(discharge.value, 0, None)
    out["soc_mwh"] = soc.value[1:]  # end-of-hour state of charge
    for k in ALL_PRODUCTS:
        out[f"r_{k}_mw"] = np.clip(reserves[k].value, 0, None)
    out["energy_revenue"] = p * (out["discharge_mw"] - out["charge_mw"])
    for k in ALL_PRODUCTS:
        out[f"as_revenue_{k}"] = prices[k].to_numpy() * out[f"r_{k}_mw"]
    out["degradation_cost"] = spec.degradation_cost_per_mwh * (
        out["charge_mw"] + out["discharge_mw"]
    )
    out["expected_profit"] = (
        out["energy_revenue"]
        + sum(out[f"as_revenue_{k}"] for k in ALL_PRODUCTS)
        - out["degradation_cost"]
    )
    return out


def _assert_physical(schedule: pd.DataFrame, spec: BatterySpec, tol: float = 1e-3) -> None:
    """Refuse to return a schedule that breaks the battery. A governed system
    validates its own decisions before anyone else has to."""
    s = schedule
    if (s["soc_mwh"] < spec.soc_min_mwh - tol).any() or (
        s["soc_mwh"] > spec.soc_max_mwh + tol
    ).any():
        raise AssertionError("SOC bounds violated")
    if (s["charge_mw"] > spec.power_mw + tol).any() or (
        s["discharge_mw"] > spec.power_mw + tol
    ).any():
        raise AssertionError("Power limit violated")
    up = sum(s[f"r_{k}_mw"] for k in UP_PRODUCTS)
    dn = sum(s[f"r_{k}_mw"] for k in DOWN_PRODUCTS)
    if (s["discharge_mw"] + up > spec.power_mw + tol).any():
        raise AssertionError("Up-reserve power headroom violated")
    if (s["charge_mw"] + dn > spec.power_mw + tol).any():
        raise AssertionError("Down-reserve power headroom violated")
    if (up / spec.eta_discharge > s["soc_mwh"] - spec.soc_min_mwh + tol).any():
        raise AssertionError("Up reserves not energy-backed")
    if (spec.eta_charge * dn > spec.soc_max_mwh - s["soc_mwh"] + tol).any():
        raise AssertionError("Down reserves exceed remaining storage room")
    if (s["charge_mw"] * s["discharge_mw"]).max() > SIMULTANEOUS_TOL_MW:
        raise AssertionError("Simultaneous charge and discharge in final schedule")


def expected_price_from_quantiles(preds: pd.DataFrame, quantiles: list[float]) -> pd.Series:
    """Collapse a quantile forecast into an expected price for the optimizer.

    Why this matters: the optimizer needs one $/MWh signal per hour, but using
    the P50 alone throws away the tails the model worked to calibrate. Weighting
    each quantile by the probability mass it represents approximates E[price]
    and lets a fat upper tail (scarcity risk) pull the dispatch signal upward —
    the quantile fan feeding directly into the money decision.
    """
    qs = np.asarray(quantiles)
    edges = np.concatenate([[0.0], (qs[1:] + qs[:-1]) / 2, [1.0]])
    weights = np.diff(edges)
    cols = [f"q{round(q * 100):02d}" for q in quantiles]
    return pd.Series(preds[cols].to_numpy() @ weights, index=preds.index, name="energy_price")


def persistence_as_prices(panel: pd.DataFrame, day_hours: pd.DatetimeIndex) -> pd.DataFrame:
    """Ancillary price signal for the optimizer: yesterday's clearing prices.

    Why this matters: AS clearing prices for delivery day D publish at ~13:30
    on D-1 — AFTER our 09:00 issue time — so using them would be leakage. The
    persistence forecast (same hour, prior day) is the strongest signal that is
    legitimately knowable. Settlement still uses the real prices in Phase 5.
    """
    lagged = day_hours - pd.Timedelta(hours=24)
    missing = lagged.difference(panel.index)
    if len(missing) > 0:
        raise ValueError(
            f"Cannot build AS persistence prices; prior-day hours missing: {missing[:3]}"
        )
    out = panel.loc[lagged, ALL_PRODUCTS].copy()
    out.index = day_hours
    return out


def run(config: dict[str, Any], args: object) -> pd.DataFrame:
    """Pipeline stage: dispatch one representative day on FORECAST prices.

    The representative day is the highest-priced delivery day available to the
    current mode — the day the whole system exists for. Emits fig06 and the
    schedule parquet; the full walk-forward loop arrives with the backtest.
    """
    import json

    from forecast_to_dispatch.config import resolve_path
    from forecast_to_dispatch.forecast.conformal import ConformalizedForecaster
    from forecast_to_dispatch.forecast.train import registry_file
    from forecast_to_dispatch.viz.dispatch_viz import fig06_dispatch_day

    fast = bool(getattr(args, "sample", False))
    processed = resolve_path(config, "processed")
    models_root = resolve_path(config, "models")
    if fast:
        models_root = models_root / "sample"

    panel = pd.read_parquet(processed / "ercot_prices.parquet")
    X = pd.read_parquet(processed / "features.parquet")
    targets = pd.read_parquet(processed / "targets.parquet")
    registry = json.loads(registry_file(models_root, fast).read_text())
    model = ConformalizedForecaster.load(models_root / registry["current"])

    # Highest-priced complete delivery day with a prior day available (for the
    # AS persistence signal).
    daily_max = targets["y_price"].groupby(targets.index.normalize()).max().sort_values()
    day = None
    for candidate in reversed(daily_max.index):
        hours = X.index[X.index.normalize() == candidate]
        if len(hours) == 24 and (hours - pd.Timedelta(hours=24)).isin(panel.index).all():
            day = candidate
            break
    if day is None:
        raise RuntimeError("No complete delivery day available for the dispatch demo")
    day_hours = X.index[X.index.normalize() == day]

    preds = model.predict_quantiles(X.loc[day_hours])
    prices = persistence_as_prices(panel, day_hours)
    prices["energy_price"] = expected_price_from_quantiles(preds, config["forecast"]["quantiles"])

    spec = BatterySpec.from_config(config)
    schedule = optimize_dispatch(prices, spec)
    check = lp_binary_check(prices, spec)

    fig06_dispatch_day(schedule, prices, spec, realized_price=panel["rtm_price"])
    schedule.to_parquet(processed / "dispatch_day_schedule.parquet")
    print(
        f"[dispatch] {day:%Y-%m-%d}: expected profit ${schedule['expected_profit'].sum():,.0f} "
        f"at forecast prices | LP overlap hours: {check['lp_overlap_hours']} "
        f"(escalated to MILP: {check['escalated']})"
    )
    return schedule


def lp_binary_check(prices: pd.DataFrame, spec: BatterySpec) -> dict[str, Any]:
    """The spec-mandated experiment: does the LP relaxation ever charge and
    discharge simultaneously on these prices, and what does exactness cost?"""
    lp = _solve(prices, spec, binaries=False)
    overlap_hours = int((lp["charge_mw"] * lp["discharge_mw"] > SIMULTANEOUS_TOL_MW).sum())
    result = {
        "lp_overlap_hours": overlap_hours,
        "lp_expected_profit": float(lp["expected_profit"].sum()),
        "escalated": overlap_hours > 0,
    }
    if overlap_hours:
        mi = _solve(prices, spec, binaries=True)
        result["milp_expected_profit"] = float(mi["expected_profit"].sum())
    return result
