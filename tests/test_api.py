"""Phase 7 acceptance: the API serves, audits, and refuses unapproved deployment."""

import json

import pytest

from forecast_to_dispatch.config import load_config, resolve_path


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    config = load_config()
    processed = resolve_path(config, "processed")
    models_root = resolve_path(config, "models")
    if (
        not (processed / "features.parquet").exists()
        or not (models_root / "registry.json").exists()
    ):
        pytest.skip("API tests need built features and a registered production model")

    from fastapi.testclient import TestClient

    from forecast_to_dispatch.serve import api

    # Redirect governance runtime state to a temp dir so tests never pollute
    # (or depend on) the real audit log and approvals.
    runtime = tmp_path_factory.mktemp("governance")

    class TestState(api._State):
        @property
        def audit_log(self):
            return runtime / "audit_log.jsonl"

    api.state = TestState()
    return TestClient(api.app)


@pytest.fixture(scope="module")
def a_day(client):
    from forecast_to_dispatch.serve import api

    return str(api.state.X.index[-24].date())


def test_health_reports_model_and_chain(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_version"].startswith("lgbm")
    assert body["audit_chain"]["ok"]


def test_forecast_returns_full_day_of_quantiles(client, a_day):
    r = client.post("/forecast", json={"day": a_day})
    assert r.status_code == 200
    q = r.json()["quantiles"]
    assert len(q) == 24
    first = next(iter(q.values()))
    assert list(first) == ["q05", "q25", "q50", "q75", "q95"]
    assert first["q05"] <= first["q50"] <= first["q95"]


def test_unknown_day_is_404(client):
    r = client.post("/forecast", json={"day": "1999-01-01"})
    assert r.status_code == 404


def test_dispatch_plans_and_audits(client, a_day):
    from forecast_to_dispatch.serve import api

    r = client.post("/dispatch", json={"day": a_day})
    assert r.status_code == 200
    body = r.json()
    assert len(body["schedule"]) == 24
    assert body["deployable"] is False
    log_lines = api.state.audit_log.read_text().strip().splitlines()
    events = [json.loads(line)["event_type"] for line in log_lines]
    assert "schedule_created" in events
    assert "api_request" in events


def test_deploy_without_approval_is_403(client, a_day):
    r = client.post("/dispatch", json={"day": a_day, "deploy": True})
    assert r.status_code == 403
    assert "NOT deployable" in r.json()["detail"]


def test_approve_then_deploy_succeeds_and_is_attributable(client, a_day):
    r = client.post("/approve", json={"day": a_day, "approver": "test-operator"})
    assert r.status_code == 200 and r.json()["approved"]

    r = client.post("/dispatch", json={"day": a_day, "deploy": True})
    assert r.status_code == 200
    body = r.json()
    assert body["deployable"] is True
    assert body["approval"]["approver"] == "test-operator"


def test_approval_requires_named_human_via_api(client, a_day):
    r = client.post("/approve", json={"day": a_day, "approver": "system"})
    assert r.status_code == 422


def test_audit_chain_still_verifies_after_traffic(client):
    from forecast_to_dispatch.governance import audit
    from forecast_to_dispatch.serve import api

    ok, detail = audit.verify_chain(api.state.audit_log)
    assert ok, detail
    records = audit.read_log(api.state.audit_log)
    assert len(records) >= 8  # every call above left a mark


def test_rollback_changes_served_model_without_restart(client, tmp_path):
    """The registry is resolved per request: a rollback re-points serving live."""
    from forecast_to_dispatch.serve import api

    models_root = resolve_path(load_config(), "models")
    registry_file = models_root / "registry.json"
    original = registry_file.read_text()
    registry = json.loads(original)
    if not registry.get("previous"):
        # simulate a previous deployment using the same artifacts
        registry["previous"] = registry["current"]
        registry_file.write_text(json.dumps(registry, indent=2))
    try:
        from forecast_to_dispatch.governance import rollback

        before = client.get("/health").json()["model_version"]
        rollback.rollback(models_root, api.state.audit_log, "test-operator", "api live-swap test")
        after = client.get("/health").json()["model_version"]
        assert after == json.loads(registry_file.read_text())["current"]
        assert before is not None
    finally:
        registry_file.write_text(original)  # restore the real registry
