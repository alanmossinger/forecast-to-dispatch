"""Phase 6 acceptance: every governance control actually controls something."""

import json

import numpy as np
import pandas as pd
import pytest

from forecast_to_dispatch.config import load_config
from forecast_to_dispatch.governance import audit, drift, hitl, model_card, risk_register, rollback


# ---------------------------------------------------------------- audit chain
def test_audit_chain_appends_and_verifies(tmp_path):
    log = tmp_path / "audit.jsonl"
    audit.append_event(log, "forecast_issued", {"day": "2024-05-08"})
    audit.append_event(log, "schedule_created", {"schedule_id": "abc"})
    audit.append_event(log, "dispatch_settled", {"settled_revenue": 123.0})
    ok, detail = audit.verify_chain(log)
    assert ok and "3 records" in detail


def test_audit_chain_detects_tampering(tmp_path):
    log = tmp_path / "audit.jsonl"
    for i in range(3):
        audit.append_event(log, "event", {"n": i})
    lines = log.read_text().strip().splitlines()
    doctored = json.loads(lines[1])
    doctored["payload"]["n"] = 999  # rewrite history
    lines[1] = json.dumps(doctored, sort_keys=True)
    log.write_text("\n".join(lines) + "\n")
    ok, detail = audit.verify_chain(log)
    assert not ok and "tampered" in detail


def test_audit_chain_detects_deletion(tmp_path):
    log = tmp_path / "audit.jsonl"
    for i in range(3):
        audit.append_event(log, "event", {"n": i})
    lines = log.read_text().strip().splitlines()
    log.write_text("\n".join([lines[0], lines[2]]) + "\n")  # delete the middle
    ok, _ = audit.verify_chain(log)
    assert not ok


# ---------------------------------------------------------------- HITL gate
def _schedule():
    idx = pd.date_range("2024-05-08", periods=24, freq="1h", tz="US/Central")
    return pd.DataFrame(
        {"charge_mw": 1.0, "discharge_mw": 2.0, "soc_mwh": 100.0, "r_regup_mw": 5.0}, index=idx
    )


def test_unapproved_schedule_is_blocked(tmp_path):
    with pytest.raises(PermissionError, match="NOT deployable"):
        hitl.require_approval(_schedule(), tmp_path / "approvals.json")


def test_approval_unblocks_exactly_that_schedule(tmp_path):
    approvals, log = tmp_path / "approvals.json", tmp_path / "audit.jsonl"
    schedule = _schedule()
    hitl.approve_schedule(schedule, "test-operator", approvals, log)
    assert hitl.require_approval(schedule, approvals)["approver"] == "test-operator"
    # one changed megawatt = a different schedule = approval no longer applies
    tampered = schedule.copy()
    tampered.iloc[0, 0] = 50.0
    with pytest.raises(PermissionError):
        hitl.require_approval(tampered, approvals)


def test_approval_requires_named_human(tmp_path):
    with pytest.raises(ValueError, match="named human"):
        hitl.approve_schedule(_schedule(), "system", tmp_path / "a.json", tmp_path / "l.jsonl")


# ---------------------------------------------------------------- rollback
def test_rollback_restores_previous_model(tmp_path):
    (tmp_path / "m_v1").mkdir()
    (tmp_path / "m_v2").mkdir()
    (tmp_path / "registry.json").write_text(
        json.dumps({"current": "m_v2", "previous": "m_v1", "models": {}})
    )
    registry = rollback.rollback(tmp_path, tmp_path / "audit.jsonl", "op", "drift alarm")
    assert registry["current"] == "m_v1" and registry["previous"] == "m_v2"
    records = audit.read_log(tmp_path / "audit.jsonl")
    assert records[-1]["event_type"] == "model_rollback"


def test_rollback_refuses_without_previous(tmp_path):
    (tmp_path / "registry.json").write_text(json.dumps({"current": "m_v1", "previous": None}))
    with pytest.raises(RuntimeError, match="no previous"):
        rollback.rollback(tmp_path, tmp_path / "audit.jsonl", "op", "test")


# ---------------------------------------------------------------- drift
def test_drift_passes_on_identical_distribution():
    rng = np.random.default_rng(42)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 2000)})
    cur = pd.DataFrame({"x": rng.normal(0, 1, 2000)})
    report = drift.drift_report(ref, cur, ["x"], psi_threshold=0.2, ks_pvalue_threshold=0.01)
    assert not report.loc["x", "drift"]


def test_drift_detects_regime_shift():
    rng = np.random.default_rng(42)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 2000)})
    cur = pd.DataFrame({"x": rng.normal(3, 2, 2000)})  # unmistakable shift
    report = drift.drift_report(ref, cur, ["x"], psi_threshold=0.2, ks_pvalue_threshold=0.01)
    assert report.loc["x", "drift"] and report.loc["x", "psi"] > 0.2


# ------------------------------------------------------- register & model card
def test_risk_register_loads_and_is_populated():
    config = load_config()
    from forecast_to_dispatch.config import REPO_ROOT

    risks = risk_register.load_register(REPO_ROOT / "governance" / "risk_register.yaml")
    assert len(risks) >= 8
    assert config is not None


def test_risk_register_rejects_malformed(tmp_path):
    bad = tmp_path / "rr.yaml"
    bad.write_text("- id: R1\n  title: no owner or ratings\n")
    with pytest.raises(ValueError, match="missing fields"):
        risk_register.load_register(bad)


def test_model_card_completeness_gate(tmp_path):
    ok, detail = model_card.check_completeness(tmp_path / "nope.md")
    assert not ok
    partial = tmp_path / "partial.md"
    partial.write_text("# Model card\n## Intended use\nstuff\n")
    ok, detail = model_card.check_completeness(partial)
    assert not ok and "missing sections" in detail


def test_generated_model_card_passes_gate():
    from forecast_to_dispatch.config import REPO_ROOT

    card = REPO_ROOT / "governance" / "model_card.md"
    if not card.exists():
        pytest.skip("model card not generated yet (run the governance stage)")
    ok, detail = model_card.check_completeness(card)
    assert ok, detail
