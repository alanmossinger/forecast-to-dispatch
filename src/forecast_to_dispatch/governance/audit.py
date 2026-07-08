"""Append-only, tamper-evident audit log for every forecast and dispatch decision.

Why this matters: when an autonomous agent allocates real capital, "what did
the system decide, when, from what inputs, under which model version?" must be
answerable years later — to an operator investigating a bad day, a risk
committee, or a regulator (EU AI Act record-keeping; NIST AI RMF 'Govern' and
'Manage' functions). Two properties are enforced in code, not policy:

1. **Append-only**: records are only ever added to the end of a JSONL file;
   there is no update or delete API.
2. **Tamper-evident**: each record embeds the SHA-256 hash of the previous
   record; editing or deleting any historical line breaks the chain, and
   ``verify_chain`` catches it. Trust becomes a property you can test.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GENESIS_HASH = "0" * 64


def _record_hash(record: dict[str, Any]) -> str:
    canonical = json.dumps({k: v for k, v in record.items() if k != "hash"}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def hash_inputs(obj: Any) -> str:
    """Stable fingerprint of a decision's inputs (frame, dict, ...)."""
    if hasattr(obj, "to_csv"):
        payload = obj.to_csv()
    else:
        payload = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def append_event(
    log_path: Path,
    event_type: str,
    payload: dict[str, Any],
    actor: str = "system",
) -> dict[str, Any]:
    """Append one immutable event. Returns the written record."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    prev_hash, seq = GENESIS_HASH, 0
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        if lines:
            last = json.loads(lines[-1])
            prev_hash, seq = last["hash"], last["seq"] + 1

    record: dict[str, Any] = {
        "seq": seq,
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "actor": actor,
        "payload": payload,
        "prev_hash": prev_hash,
    }
    record["hash"] = _record_hash(record)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def read_log(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").strip().splitlines()]


def verify_chain(log_path: Path) -> tuple[bool, str]:
    """Walk the hash chain; any edited, deleted, or reordered record breaks it."""
    records = read_log(log_path)
    if not records:
        return True, "empty log"
    prev = GENESIS_HASH
    for rec in records:
        if rec["prev_hash"] != prev:
            return False, f"chain broken at seq {rec['seq']}: prev_hash mismatch"
        if _record_hash(rec) != rec["hash"]:
            return False, f"record {rec['seq']} content does not match its hash (tampered)"
        prev = rec["hash"]
    return True, f"{len(records)} records verified"
