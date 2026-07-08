"""Risk register: schema-validated source of truth, rendered for humans.

Why this matters: a risk register that lives in a slide deck is decoration; one
that is schema-checked in CI and versioned with the code is governance. The
YAML is the source of truth; this module refuses malformed entries (missing
owner, invalid rating) and renders the markdown a reviewer actually reads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_FIELDS = {
    "id",
    "title",
    "description",
    "likelihood",
    "impact",
    "mitigation",
    "owner",
    "status",
}
VALID_RATINGS = {"low", "medium", "high"}


def load_register(path: Path) -> list[dict[str, Any]]:
    """Load and validate; malformed risk entries fail loudly (a silent gap in
    the register is itself a risk)."""
    if not path.exists():
        raise FileNotFoundError(f"Risk register not found: {path}")
    risks = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(risks, list) or not risks:
        raise ValueError("Risk register must be a non-empty list of risks")
    ids = set()
    for r in risks:
        missing = REQUIRED_FIELDS - set(r)
        if missing:
            raise ValueError(f"Risk {r.get('id', '?')} missing fields: {sorted(missing)}")
        if r["likelihood"] not in VALID_RATINGS or r["impact"] not in VALID_RATINGS:
            raise ValueError(f"Risk {r['id']}: ratings must be one of {sorted(VALID_RATINGS)}")
        if r["id"] in ids:
            raise ValueError(f"Duplicate risk id {r['id']}")
        ids.add(r["id"])
    return risks


def render_markdown(risks: list[dict[str, Any]], out_path: Path) -> Path:
    rank = {"low": 0, "medium": 1, "high": 2}
    risks_sorted = sorted(risks, key=lambda r: -(rank[r["likelihood"]] + rank[r["impact"]]))
    lines = [
        "# Risk register — Forecast-to-Dispatch",
        "",
        "*Generated from `governance/risk_register.yaml` (source of truth). "
        "Schema-validated in CI; a malformed or empty register fails the release gate.*",
        "",
        "| ID | Risk | Likelihood | Impact | Status | Owner |",
        "|---|---|---|---|---|---|",
    ]
    for r in risks_sorted:
        lines.append(
            f"| {r['id']} | {r['title']} | {r['likelihood']} | {r['impact']} "
            f"| {r['status']} | {r['owner']} |"
        )
    lines.append("")
    for r in risks_sorted:
        lines += [
            f"## {r['id']} — {r['title']}",
            "",
            f"**Likelihood:** {r['likelihood']} · **Impact:** {r['impact']} · "
            f"**Status:** {r['status']} · **Owner:** {r['owner']}",
            "",
            f"{r['description'].strip()}",
            "",
            f"**Mitigation:** {r['mitigation'].strip()}",
            "",
        ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
