"""Serving API: the operator's interface, with governance wired into every request.

Why this matters: this is the surface a production operator (or downstream
bidding system) would actually call. Three governance properties are enforced
*in the transport layer*, not left to caller discipline:

1. **Every request is audited** — middleware appends each call (path, status,
   duration, payload hash) to the same append-only hash-chained log the
   decision events use.
2. **Deployment requires a human** — ``/dispatch`` happily *plans* a schedule,
   but asking for ``deploy=true`` without a logged approval of exactly that
   schedule returns HTTP 403. Approval happens via ``/approve`` with a named
   approver, and is bound to the schedule's content hash.
3. **The model is resolved from the registry at request time** — a rollback
   re-points the registry and the very next request serves the previous
   model; no redeploy, no restart.

Run locally:
    .venv/Scripts/uvicorn forecast_to_dispatch.serve.api:app --port 8000
    curl http://127.0.0.1:8000/health
"""

from __future__ import annotations

import json
import time
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from forecast_to_dispatch.config import load_config, resolve_path
from forecast_to_dispatch.forecast.conformal import ConformalizedForecaster
from forecast_to_dispatch.governance import audit, hitl
from forecast_to_dispatch.optimize.dispatch import (
    BatterySpec,
    expected_price_from_quantiles,
    optimize_dispatch,
    persistence_as_prices,
)

app = FastAPI(
    title="Forecast-to-Dispatch",
    description="Governed price forecasting and battery dispatch (ERCOT).",
)


class _State:
    """Lazy-loaded serving state; the model is re-resolved from the registry
    on every access so rollback takes effect immediately."""

    def __init__(self) -> None:
        self.config = load_config()
        self._X: pd.DataFrame | None = None
        self._panel: pd.DataFrame | None = None

    @property
    def audit_log(self):
        return resolve_path(self.config, "audit_log")

    @property
    def approvals(self):
        return self.audit_log.parent / "approvals.json"

    @property
    def X(self) -> pd.DataFrame:
        if self._X is None:
            self._X = pd.read_parquet(resolve_path(self.config, "processed") / "features.parquet")
        return self._X

    @property
    def panel(self) -> pd.DataFrame:
        if self._panel is None:
            self._panel = pd.read_parquet(
                resolve_path(self.config, "processed") / "ercot_prices.parquet"
            )
        return self._panel

    def current_model(self) -> tuple[str, ConformalizedForecaster]:
        models_root = resolve_path(self.config, "models")
        registry = json.loads((models_root / "registry.json").read_text())
        return registry["current"], ConformalizedForecaster.load(models_root / registry["current"])

    def day_hours(self, day: str) -> pd.DatetimeIndex:
        tz = self.config["market"]["timezone"]
        d = pd.Timestamp(day, tz=tz).normalize()
        hours = self.X.index[self.X.index.normalize() == d]
        if len(hours) != 24:
            raise HTTPException(
                status_code=404,
                detail=f"No complete feature day for {day} (found {len(hours)} hours). "
                "Run the ingest+features stages for a window covering it.",
            )
        return hours


state = _State()


@app.middleware("http")
async def audit_every_request(request: Request, call_next):
    """Nothing touches this service off the record."""
    started = time.perf_counter()
    body = await request.body()
    response = await call_next(request)
    audit.append_event(
        state.audit_log,
        "api_request",
        {
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": round(1000 * (time.perf_counter() - started), 1),
            "body_hash": audit.hash_inputs(body.decode() or "{}"),
        },
        actor=request.client.host if request.client else "unknown",
    )
    return response


class ForecastRequest(BaseModel):
    day: str = Field(..., description="Delivery day, YYYY-MM-DD (must be in the feature window)")


class DispatchRequest(BaseModel):
    day: str
    deploy: bool = Field(
        False,
        description="True = request a DEPLOYABLE schedule; refused (403) without "
        "a logged human approval of exactly this schedule.",
    )


class ApproveRequest(BaseModel):
    day: str
    approver: str = Field(..., description="Named human approver (never 'system')")
    note: str = ""


def _plan_schedule(day: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    hours = state.day_hours(day)
    version, model = state.current_model()
    preds = model.predict_quantiles(state.X.loc[hours])
    prices = persistence_as_prices(state.panel, hours)
    prices["energy_price"] = expected_price_from_quantiles(
        preds, state.config["forecast"]["quantiles"]
    )
    schedule = optimize_dispatch(prices, BatterySpec.from_config(state.config))
    return schedule, preds, version


@app.get("/health")
def health() -> dict[str, Any]:
    version, _ = state.current_model()
    chain_ok, chain_detail = audit.verify_chain(state.audit_log)
    return {
        "status": "ok",
        "model_version": version,
        "audit_chain": {"ok": chain_ok, "detail": chain_detail},
        "feature_window": [str(state.X.index.min()), str(state.X.index.max())],
    }


@app.post("/forecast")
def forecast(req: ForecastRequest) -> dict[str, Any]:
    hours = state.day_hours(req.day)
    version, model = state.current_model()
    preds = model.predict_quantiles(state.X.loc[hours])
    audit.append_event(
        state.audit_log,
        "forecast_issued",
        {
            "day": req.day,
            "model_version": version,
            "input_hash": audit.hash_inputs(state.X.loc[hours]),
            "p95_max": float(preds["q95"].max()),
        },
        actor="api",
    )
    return {
        "day": req.day,
        "model_version": version,
        "quantiles": {ts.isoformat(): row.to_dict() for ts, row in preds.round(2).iterrows()},
    }


@app.post("/dispatch")
def dispatch(req: DispatchRequest) -> dict[str, Any]:
    schedule, preds, version = _plan_schedule(req.day)
    sid = hitl.schedule_id(schedule)

    deployable = False
    approval: dict[str, Any] | None = None
    if req.deploy:
        try:
            approval = hitl.require_approval(schedule, state.approvals)
            deployable = True
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    audit.append_event(
        state.audit_log,
        "schedule_created",
        {
            "day": req.day,
            "model_version": version,
            "schedule_id": sid,
            "expected_profit": float(schedule["expected_profit"].sum()),
            "deployable": deployable,
        },
        actor="api",
    )
    return {
        "day": req.day,
        "model_version": version,
        "schedule_id": sid,
        "deployable": deployable,
        "approval": approval,
        "expected_profit": round(float(schedule["expected_profit"].sum()), 2),
        "schedule": {ts.isoformat(): row.to_dict() for ts, row in schedule.round(3).iterrows()},
    }


@app.post("/approve")
def approve(req: ApproveRequest) -> dict[str, Any]:
    schedule, _, _ = _plan_schedule(req.day)
    try:
        record = hitl.approve_schedule(
            schedule, req.approver, state.approvals, state.audit_log, note=req.note
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"approved": True, **record}
