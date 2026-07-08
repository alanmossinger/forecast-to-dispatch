"""The release gate: one command, every governance control, non-zero on failure.

Why this matters: governance that depends on someone remembering to check is
not governance. This module is the single command CI runs
(``python -m forecast_to_dispatch.governance.check``); if ANY gate fails —
missing model card, empty risk register, dishonest coverage, corroborated
drift, broken audit chain, implausible backtest — the process exits non-zero
and an ungoverned model literally cannot merge or ship.

Gates (each returns PASS/FAIL + detail, all reported, first failure does not
short-circuit the rest):
  G1 model card complete          G2 risk register valid & non-empty
  G3 interval coverage honest     G4 scarcity recall reported
  G5 leakage spec intact          G6 drift within thresholds
  G7 audit chain verifies         G8 backtest capture plausible
  G9 registry consistent
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from forecast_to_dispatch.config import load_config, resolve_path
from forecast_to_dispatch.governance import audit, drift, model_card, risk_register

Gate = tuple[str, str, Callable[[], tuple[bool, str]]]


def _paths(config: dict[str, Any], sample: bool) -> dict[str, Path]:
    from forecast_to_dispatch.forecast.train import registry_file

    models_root = resolve_path(config, "models")
    if sample:
        models_root = models_root / "sample"
    return {
        "models_root": models_root,
        "registry": registry_file(models_root, sample),
        "processed": resolve_path(config, "processed"),
        "card": resolve_path(config, "models").parent / "governance" / "model_card.md",
        "register": resolve_path(config, "models").parent / "governance" / "risk_register.yaml",
        "audit_log": resolve_path(config, "audit_log"),
    }


def build_gates(config: dict[str, Any], p: dict[str, Path]) -> list[Gate]:
    def g_model_card() -> tuple[bool, str]:
        return model_card.check_completeness(p["card"])

    def g_risk_register() -> tuple[bool, str]:
        risks = risk_register.load_register(p["register"])
        return True, f"{len(risks)} risks, schema valid"

    def _current_metrics() -> dict[str, Any]:
        registry = json.loads(p["registry"].read_text())
        return json.loads((p["models_root"] / registry["current"] / "metrics.json").read_text())[
            "models"
        ]["lgbm"]

    def g_coverage() -> tuple[bool, str]:
        cov = _current_metrics()["coverage_90"]
        tol = config["forecast"]["coverage_tolerance"]
        ok = abs(cov - 0.90) <= tol
        return ok, f"coverage_90={cov:.3f}, tolerance ±{tol} around 0.90"

    def g_scarcity() -> tuple[bool, str]:
        m = _current_metrics()
        r = m["scarcity_recall"]
        if m["scarcity_hours"] == 0:
            return True, "no scarcity hours in evaluation window (reported, not hidden)"
        ok = r is not None and not math.isnan(r)
        return ok, f"scarcity_recall={r:.2f} on {m['scarcity_hours']} hours"

    def g_leakage() -> tuple[bool, str]:
        from forecast_to_dispatch.features.build import FEATURE_SPEC, MIN_LAG_BY_SOURCE

        bad = [n for n, r in FEATURE_SPEC.items() if r.lag_hours < MIN_LAG_BY_SOURCE[r.source]]
        return (not bad), ("all declared lags legal" if not bad else f"illegal lags: {bad}")

    def g_drift() -> tuple[bool, str]:
        import yaml

        X = pd.read_parquet(p["processed"] / "features.parquet")
        registry = json.loads(p["registry"].read_text())
        metrics = json.loads((p["models_root"] / registry["current"] / "metrics.json").read_text())
        train_end = pd.Timestamp(metrics["train_window"][1])
        reference = X.loc[X.index <= train_end]
        current = X.loc[X.index > train_end]
        if len(current) < 48:
            return True, "insufficient post-training data for drift (needs 48h)"
        current = current.iloc[-14 * 24 :]
        report = drift.drift_report(
            reference,
            current,
            [c for c in drift.MONITORED_FEATURES if c in X.columns],
            config["governance"]["drift"]["psi_threshold"],
            config["governance"]["drift"]["ks_pvalue_threshold"],
        )
        drifted = report.index[report["drift"]].tolist()
        detail = ", ".join(f"{i}: psi={r.psi:.2f}" for i, r in report.iterrows())
        if not drifted:
            return True, detail

        # Corroborated drift: PASS only if a named owner acknowledged exactly
        # this (series covered + window not expired). Detect -> human decision
        # -> documented acceptance; an acknowledgment expires, it is not a mute.
        ack_file = p["register"].parent / "drift_acknowledgment.yaml"
        if ack_file.exists():
            acks = yaml.safe_load(ack_file.read_text(encoding="utf-8")) or []
            window_max = current.index.max().tz_localize(None)
            for ack in acks:
                covered = set(drifted) <= set(ack.get("series", []))
                fresh = pd.Timestamp(ack.get("window_end", "1970-01-01")) >= window_max
                if covered and fresh:
                    return True, (
                        f"drift DETECTED in {drifted} and ACKNOWLEDGED by "
                        f"{ack['acknowledged_by']} (risk {ack.get('linked_risk')}, "
                        f"valid through {ack['window_end']}) | {detail}"
                    )
        return False, f"UNACKNOWLEDGED drift in {drifted} | {detail}"

    def g_audit() -> tuple[bool, str]:
        return audit.verify_chain(p["audit_log"])

    def g_backtest() -> tuple[bool, str]:
        f = p["processed"] / "backtest_results.json"
        if not f.exists():
            return False, "backtest_results.json missing — run the backtest stage"
        s = json.loads(f.read_text())
        cap = s["revenue_capture_pct"]
        ok = 20.0 < cap < 100.0 and s["totals"]["governed"] > s["totals"]["naive"]
        return (
            ok,
            f"capture={cap:.1f}%, governed>naive={s['totals']['governed'] > s['totals']['naive']}",
        )

    def g_registry() -> tuple[bool, str]:
        reg_file = p["registry"]
        if not reg_file.exists():
            return False, f"{reg_file.name} missing"
        registry = json.loads(reg_file.read_text())
        cur = p["models_root"] / registry["current"]
        needed = [cur / "metrics.json", cur / "config_snapshot.yaml", cur / "conformal.json"]
        missing = [str(f.name) for f in needed if not f.exists()]
        return (not missing), (
            f"current={registry['current']}" if not missing else f"missing: {missing}"
        )

    return [
        ("G1", "model card complete", g_model_card),
        ("G2", "risk register valid", g_risk_register),
        ("G3", "interval coverage honest", g_coverage),
        ("G4", "scarcity recall reported", g_scarcity),
        ("G5", "leakage spec intact", g_leakage),
        ("G6", "drift within thresholds", g_drift),
        ("G7", "audit chain verifies", g_audit),
        ("G8", "backtest capture plausible", g_backtest),
        ("G9", "model registry consistent", g_registry),
    ]


def run_gates(config: dict[str, Any], sample: bool = False) -> tuple[bool, list[dict[str, Any]]]:
    p = _paths(config, sample)
    results = []
    for gate_id, name, fn in build_gates(config, p):
        try:
            ok, detail = fn()
        except Exception as exc:  # a crashing gate is a failing gate
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        results.append({"gate": gate_id, "name": name, "passed": bool(ok), "detail": str(detail)})
    all_pass = all(r["passed"] for r in results)
    return all_pass, results


def refresh_artifacts(config: dict[str, Any], sample: bool = False) -> None:
    """Regenerate the generated governance artifacts from current run outputs."""
    p = _paths(config, sample)
    backtest = None
    bt_file = p["processed"] / "backtest_results.json"
    if bt_file.exists():
        backtest = json.loads(bt_file.read_text())
    model_card.generate_model_card(
        config, p["models_root"], backtest, p["card"], registry_path=p["registry"]
    )
    risks = risk_register.load_register(p["register"])
    risk_register.render_markdown(risks, p["register"].with_suffix(".md"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true", help="gate the sample/CI artifacts")
    parser.add_argument(
        "--refresh-artifacts",
        action="store_true",
        help="regenerate model card + risk markdown first",
    )
    args = parser.parse_args()
    config = load_config()
    if args.refresh_artifacts:
        refresh_artifacts(config, sample=args.sample)

    all_pass, results = run_gates(config, sample=args.sample)
    width = max(len(r["name"]) for r in results)
    print("\n=== GOVERNANCE RELEASE GATES ===")
    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"  {r['gate']}  {r['name']:<{width}}  [{mark}]  {r['detail']}")
    print(
        f"=== {'ALL GATES PASS — release permitted' if all_pass else 'GATE FAILURE — release blocked'} ===\n"
    )

    gate_status = resolve_path(config, "models").parent / "governance" / "gate_status.json"
    gate_status.write_text(json.dumps(results, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
