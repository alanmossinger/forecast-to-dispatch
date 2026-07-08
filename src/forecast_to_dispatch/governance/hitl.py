"""Human-in-the-loop gate: no dispatch schedule is deployable without a person.

Why this matters: this is the oversight requirement of both NIST AI RMF
(human oversight of consequential automated decisions) and the EU AI Act made
*mechanical*. A schedule is identified by the hash of its content; approval is
a logged, attributable act tied to that exact hash. Change one number in the
schedule and the approval no longer applies. The API (Phase 7) and CI both
refuse unapproved schedules — the gate cannot be talked around.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from forecast_to_dispatch.governance.audit import append_event, hash_inputs


def schedule_id(schedule: pd.DataFrame) -> str:
    """The schedule IS its content: any modification yields a new identity."""
    core_cols = [c for c in schedule.columns if c.endswith("_mw") or c == "soc_mwh"]
    return hash_inputs(schedule[core_cols].round(6))


def approve_schedule(
    schedule: pd.DataFrame,
    approver: str,
    approvals_path: Path,
    audit_log_path: Path,
    note: str = "",
) -> dict[str, Any]:
    """Record a human approval for exactly this schedule, and audit it."""
    if not approver or approver == "system":
        raise ValueError("Approval requires a named human approver")
    sid = schedule_id(schedule)
    approvals = _load(approvals_path)
    record = {
        "schedule_id": sid,
        "approver": approver,
        "note": note,
        "day": str(schedule.index[0].date()) if len(schedule) else None,
    }
    audit_rec = append_event(audit_log_path, "hitl_approval", record, actor=approver)
    record["ts_utc"] = audit_rec["ts_utc"]
    approvals[sid] = record
    approvals_path.parent.mkdir(parents=True, exist_ok=True)
    approvals_path.write_text(json.dumps(approvals, indent=2))
    return record


def is_approved(schedule: pd.DataFrame, approvals_path: Path) -> bool:
    return schedule_id(schedule) in _load(approvals_path)


def require_approval(schedule: pd.DataFrame, approvals_path: Path) -> dict[str, Any]:
    """The gate itself: raises unless a human approved exactly this schedule."""
    approvals = _load(approvals_path)
    sid = schedule_id(schedule)
    if sid not in approvals:
        raise PermissionError(
            f"Schedule {sid[:12]}... is NOT deployable: no human approval on record. "
            "A dispatch schedule requires a logged approval before deployment."
        )
    return approvals[sid]


def _load(approvals_path: Path) -> dict[str, Any]:
    if not approvals_path.exists():
        return {}
    return json.loads(approvals_path.read_text())
