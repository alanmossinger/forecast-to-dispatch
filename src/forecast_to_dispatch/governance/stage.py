"""Governance pipeline stage: exercise every control on the real artifacts.

Why this matters: governance components that are never *run* rot. This stage
regenerates the model card and risk register from the current run, writes a
real decision sequence into the audit log (forecast → schedule → human
approval → settlement), computes the drift picture, runs all release gates,
and emits fig11/fig12/fig13/fig24. If any gate fails, the stage raises and the
pipeline is red — same behavior CI enforces.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from forecast_to_dispatch.config import resolve_path
from forecast_to_dispatch.governance import audit, check, drift, hitl
from forecast_to_dispatch.viz.architecture import fig13_architecture
from forecast_to_dispatch.viz.governance_viz import (
    fig11_drift_monitor,
    fig12_gate_status,
    fig24_audit_trail_flow,
)


def _seed_decision_trail(config: dict[str, Any], sample: bool) -> None:
    """Record one real decision sequence in the audit log (idempotent-ish:
    skipped if a settlement event already exists)."""
    from forecast_to_dispatch.forecast.conformal import ConformalizedForecaster
    from forecast_to_dispatch.forecast.train import registry_file

    processed = resolve_path(config, "processed")
    models_root = resolve_path(config, "models")
    if sample:
        models_root = models_root / "sample"
    log_path = resolve_path(config, "audit_log")
    approvals_path = log_path.parent / "approvals.json"

    existing = audit.read_log(log_path)
    if any(r["event_type"] == "dispatch_settled" for r in existing):
        return

    registry = json.loads(registry_file(models_root, sample).read_text())
    model = ConformalizedForecaster.load(models_root / registry["current"])
    X = pd.read_parquet(processed / "features.parquet")
    schedule = pd.read_parquet(processed / "dispatch_day_schedule.parquet")
    day = schedule.index[0].normalize()
    day_hours = X.index[X.index.normalize() == day]
    preds = model.predict_quantiles(X.loc[day_hours])

    audit.append_event(
        log_path,
        "forecast_issued",
        {
            "day": str(day.date()),
            "model_version": registry["current"],
            "input_hash": audit.hash_inputs(X.loc[day_hours]),
            "p95_max": float(preds["q95"].max()),
        },
    )
    audit.append_event(
        log_path,
        "schedule_created",
        {
            "day": str(day.date()),
            "schedule_id": hitl.schedule_id(schedule),
            "expected_profit": float(schedule["expected_profit"].sum()),
        },
    )
    # Demo of the mechanism: in production this is a person in the loop; the
    # scripted approver is clearly labeled as such.
    hitl.approve_schedule(
        schedule,
        approver="demo-operator",
        approvals_path=approvals_path,
        audit_log_path=log_path,
        note="walkthrough approval — mechanism demo",
    )
    daily_file = processed / "backtest_daily.parquet"
    settled = None
    if daily_file.exists():
        daily = pd.read_parquet(daily_file)
        if day in daily.index:
            settled = float(daily.loc[day, "governed_revenue"])
    audit.append_event(
        log_path,
        "dispatch_settled",
        {"day": str(day.date()), "settled_revenue": settled, "approved": True},
    )


def run(config: dict[str, Any], args: object) -> list[dict[str, Any]]:
    sample = bool(getattr(args, "sample", False))
    processed = resolve_path(config, "processed")
    models_root = resolve_path(config, "models")
    if sample:
        models_root = models_root / "sample"

    # 1. Regenerate generated artifacts from the current run.
    check.refresh_artifacts(config, sample=sample)

    # 2. Record a real decision sequence in the audit trail.
    _seed_decision_trail(config, sample)

    # 3. Drift picture: weekly PSI of monitored features + the model's own P50.
    from forecast_to_dispatch.forecast.conformal import ConformalizedForecaster
    from forecast_to_dispatch.forecast.train import registry_file

    X = pd.read_parquet(processed / "features.parquet")
    registry = json.loads(registry_file(models_root, sample).read_text())
    model = ConformalizedForecaster.load(models_root / registry["current"])
    metrics = json.loads((models_root / registry["current"] / "metrics.json").read_text())
    train_end = pd.Timestamp(metrics["train_window"][1])

    monitored = [c for c in drift.MONITORED_FEATURES if c in X.columns]
    frame = X[monitored].copy()
    frame["prediction_q50"] = model.predict_quantiles(X)["q50"]
    reference = frame.loc[frame.index <= train_end]
    stream = frame.loc[frame.index > train_end]
    rolling = drift.rolling_psi(reference, stream, list(frame.columns))
    fig11_drift_monitor(rolling, config["governance"]["drift"]["psi_threshold"])

    # 4. Architecture + audit-trail figures.
    fig13_architecture()
    fig24_audit_trail_flow(audit.read_log(resolve_path(config, "audit_log")))

    # 5. Run every release gate; fig12 shows the verdicts; failure = red pipeline.
    all_pass, results = check.run_gates(config, sample=sample)
    fig12_gate_status(results)
    gate_status = resolve_path(config, "models").parent / "governance" / "gate_status.json"
    gate_status.write_text(json.dumps(results, indent=2))
    for r in results:
        print(
            f"[governance] {r['gate']} {r['name']}: {'PASS' if r['passed'] else 'FAIL'} — {r['detail']}"
        )
    if not all_pass:
        raise RuntimeError("Governance gate failure — release blocked (see gate_status.json)")
    print("[governance] ALL GATES PASS — release permitted")
    return results
