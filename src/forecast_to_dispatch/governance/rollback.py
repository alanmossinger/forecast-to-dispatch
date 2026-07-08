"""Model rollback: one command to re-point serving at the previous version.

Why this matters: when the drift monitor fires or a retrain degrades, the
question is not "can we fix it?" but "how fast can we get back to the last
known-good model?" The registry keeps `current` and `previous` pointers; this
module swaps them atomically, audits the act, and refuses to roll back to
nothing. The serving API reads the registry at request time, so a rollback
takes effect without redeploying code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from forecast_to_dispatch.governance.audit import append_event


def rollback(models_root: Path, audit_log_path: Path, actor: str, reason: str) -> dict[str, Any]:
    """Swap current <-> previous in the registry. Returns the new registry."""
    registry_path = models_root / "registry.json"
    registry = json.loads(registry_path.read_text())
    previous = registry.get("previous")
    if not previous:
        raise RuntimeError("Rollback refused: no previous model version in the registry")
    if not (models_root / previous).exists():
        raise RuntimeError(f"Rollback refused: previous model artifacts missing ({previous})")

    old_current = registry["current"]
    registry["current"], registry["previous"] = previous, old_current
    registry_path.write_text(json.dumps(registry, indent=2))
    append_event(
        audit_log_path,
        "model_rollback",
        {"from": old_current, "to": previous, "reason": reason},
        actor=actor,
    )
    return registry
