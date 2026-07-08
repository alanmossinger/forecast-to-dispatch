"""Walk-forward backtest: decide on the forecast, settle on reality.

Why this matters: this file produces the numbers a hiring manager or asset
owner will quote — total revenue, revenue capture vs the perfect-foresight
ceiling, uplift vs a naive floor. Its honesty rests on one mechanism enforced
here and nowhere else: **the dispatch decision is made on forecast prices;
the revenue is settled at the prices that actually occurred.** Optimizing and
settling on the same forecast is how storage studies overstate profit by 20%+.

Three policies, same battery, same physics, every delivery day of the
out-of-sample window:

- ``perfect``  — optimizer sees realized prices (energy + AS). An unreachable
  upper bound: nobody knows tomorrow's prices. Its role is honest scaling.
- ``naive``    — optimizer sees the *training-window average day* (hour-of-day
  mean RTM price) and no ancillary prices: energy-only, no forecast, no model.
  The floor any strategy must beat.
- ``governed`` — optimizer sees the conformal quantile forecast (expected
  price from the calibrated fan) plus persistence AS prices. This is the
  system under evaluation.

Protocol notes (stated in the model card):
- The forecaster is trained once at the study cutoff; every backtest day is
  strictly after its training window. (A production system would retrain on a
  schedule; a fixed model is the conservative choice here.)
- Days are independent with a cyclic state of charge (the battery must end
  each day where it started — no free energy carried into the report).
- Reserve awards settle at realized clearing prices on offered capacity
  (price-taker assumption, consistent with dispatch).
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from forecast_to_dispatch.optimize.dispatch import (
    ALL_PRODUCTS,
    BatterySpec,
    expected_price_from_quantiles,
    optimize_dispatch,
    persistence_as_prices,
)

REVENUE_STREAMS = ["energy"] + ALL_PRODUCTS


def settle_on_realized_prices(
    schedule: pd.DataFrame, realized: pd.DataFrame, spec: BatterySpec
) -> pd.DataFrame:
    """Revalue a dispatch schedule at the prices that actually occurred.

    Why this matters: optimizing and settling on the same forecast inflates
    revenue because the model 'knows' the future it invented. Settling on
    realized prices is what makes the reported revenue capture defensible to
    an operator — the schedule was committed before reality arrived.
    """
    out = pd.DataFrame(index=schedule.index)
    out["energy"] = realized["rtm_price"] * (schedule["discharge_mw"] - schedule["charge_mw"])
    for k in ALL_PRODUCTS:
        out[k] = realized[k] * schedule[f"r_{k}_mw"]
    out["degradation"] = -spec.degradation_cost_per_mwh * (
        schedule["charge_mw"] + schedule["discharge_mw"]
    )
    out["net_revenue"] = out[REVENUE_STREAMS].sum(axis=1) + out["degradation"]
    return out


def climatology_price_curve(panel_train: pd.DataFrame) -> pd.Series:
    """The naive policy's entire worldview: the average day of the past."""
    return panel_train["rtm_price"].groupby(panel_train.index.hour).mean()


def run_backtest(
    panel: pd.DataFrame,
    X: pd.DataFrame,
    model,
    quantiles: list[float],
    spec: BatterySpec,
    train_end: pd.Timestamp,
) -> dict[str, Any]:
    """Walk forward over every complete delivery day after ``train_end``."""
    test_days = [
        d
        for d in X.index.normalize().unique().sort_values()
        if d > train_end
        and (X.index.normalize() == d).sum() == 24
        and (X.index[X.index.normalize() == d] - pd.Timedelta(hours=24)).isin(panel.index).all()
    ]
    if not test_days:
        raise ValueError("No complete out-of-sample delivery days to backtest")

    clim_curve = climatology_price_curve(panel.loc[panel.index.normalize() <= train_end])

    daily_rows: list[dict[str, Any]] = []
    schedules: dict[str, list[pd.DataFrame]] = {"governed": [], "perfect": [], "naive": []}

    for day in test_days:
        hours = X.index[X.index.normalize() == day]
        realized = panel.loc[hours]

        # --- governed: forecast fan -> expected price; persistence AS prices
        preds = model.predict_quantiles(X.loc[hours])
        gov_prices = persistence_as_prices(panel, hours)
        gov_prices["energy_price"] = expected_price_from_quantiles(preds, quantiles)
        gov_schedule = optimize_dispatch(gov_prices, spec)

        # --- perfect foresight: realized energy AND realized AS prices
        pf_prices = realized[ALL_PRODUCTS].copy()
        pf_prices["energy_price"] = realized["rtm_price"]
        pf_schedule = optimize_dispatch(pf_prices, spec)

        # --- naive: the average training day, energy only
        nv_prices = pd.DataFrame(index=hours)
        nv_prices["energy_price"] = [clim_curve[h] for h in hours.hour]
        for k in ALL_PRODUCTS:
            nv_prices[k] = 0.0
        nv_schedule = optimize_dispatch(nv_prices, spec)

        row: dict[str, Any] = {"day": day}
        for name, schedule in (
            ("governed", gov_schedule),
            ("perfect", pf_schedule),
            ("naive", nv_schedule),
        ):
            settled = settle_on_realized_prices(schedule, realized, spec)
            row[f"{name}_revenue"] = settled["net_revenue"].sum()
            if name == "governed":
                for stream in REVENUE_STREAMS + ["degradation"]:
                    row[f"governed_{stream}"] = settled[stream].sum()
            schedules[name].append(schedule.assign(day=day))
        row["forecast_mae"] = (realized["rtm_price"] - preds["q50"]).abs().mean()
        row["realized_max_price"] = realized["rtm_price"].max()
        daily_rows.append(row)

    daily = pd.DataFrame(daily_rows).set_index("day")
    totals = {p: float(daily[f"{p}_revenue"].sum()) for p in ("governed", "perfect", "naive")}
    results = {
        "n_days": len(daily),
        "window": [str(test_days[0].date()), str(test_days[-1].date())],
        "totals": totals,
        "revenue_capture_pct": 100 * totals["governed"] / totals["perfect"],
        "uplift_vs_naive_pct": 100 * (totals["governed"] / totals["naive"] - 1),
        "governed_streams": {
            s: float(daily[f"governed_{s}"].sum()) for s in REVENUE_STREAMS + ["degradation"]
        },
    }
    return {
        "summary": results,
        "daily": daily,
        "governed_schedule": pd.concat(schedules["governed"]),
        "perfect_schedule": pd.concat(schedules["perfect"]),
        "naive_schedule": pd.concat(schedules["naive"]),
    }


def run(config: dict[str, Any], args: object) -> dict[str, Any]:
    """Pipeline stage: the full walk-forward backtest + persisted artifacts."""
    from forecast_to_dispatch.config import resolve_path
    from forecast_to_dispatch.forecast.conformal import ConformalizedForecaster
    from forecast_to_dispatch.forecast.train import registry_file

    fast = bool(getattr(args, "sample", False))
    processed = resolve_path(config, "processed")
    models_root = resolve_path(config, "models")
    if fast:
        models_root = models_root / "sample"

    panel = pd.read_parquet(processed / "ercot_prices.parquet")
    X = pd.read_parquet(processed / "features.parquet")
    registry = json.loads(registry_file(models_root, fast).read_text())
    model = ConformalizedForecaster.load(models_root / registry["current"])
    metrics = json.loads((models_root / registry["current"] / "metrics.json").read_text())
    train_end = pd.Timestamp(metrics["train_window"][1]).normalize()

    result = run_backtest(
        panel,
        X,
        model,
        config["forecast"]["quantiles"],
        BatterySpec.from_config(config),
        train_end,
    )

    result["daily"].to_parquet(processed / "backtest_daily.parquet")
    result["governed_schedule"].to_parquet(processed / "backtest_governed_schedule.parquet")
    (processed / "backtest_results.json").write_text(json.dumps(result["summary"], indent=2))

    s = result["summary"]
    print(
        f"[backtest] {s['n_days']} days ({s['window'][0]} .. {s['window'][1]}): "
        f"governed ${s['totals']['governed']:,.0f} | naive ${s['totals']['naive']:,.0f} | "
        f"perfect ${s['totals']['perfect']:,.0f} | capture {s['revenue_capture_pct']:.1f}% | "
        f"uplift vs naive {s['uplift_vs_naive_pct']:+.1f}%"
    )
    return result
