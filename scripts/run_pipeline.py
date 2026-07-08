"""End-to-end pipeline: ingest -> features -> forecast -> dispatch -> backtest -> governance.

Why this matters: one command that reruns the whole decision loop from raw data
to governed result is what makes the system auditable and reproducible. Stages
are registered here as their phases land; asking for a stage that is not built
yet fails loudly instead of pretending.

Usage:
    python scripts/run_pipeline.py --sample            # offline, committed sample
    python scripts/run_pipeline.py --start 2024-01-01 --end 2024-06-30
"""

from __future__ import annotations

import argparse
import sys

from forecast_to_dispatch.config import load_config
from forecast_to_dispatch.data import ingest
from forecast_to_dispatch.features import build as features_build
from forecast_to_dispatch.backtest import engine as backtest_engine
from forecast_to_dispatch.forecast import train as forecast_train
from forecast_to_dispatch.governance import stage as governance_stage
from forecast_to_dispatch.optimize import dispatch as optimize_dispatch_stage

# Stages register here phase by phase: name -> callable(config, args)
STAGES: dict[str, object] = {
    "ingest": ingest.run,
    "features": features_build.run,
    "forecast": forecast_train.run,
    "dispatch": optimize_dispatch_stage.run,
    "backtest": backtest_engine.run,
    "governance": governance_stage.run,
}

PLANNED_STAGES = ["ingest", "features", "forecast", "dispatch", "backtest", "governance"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true", help="run offline from committed sample")
    parser.add_argument("--start", help="live-data start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="live-data end date (YYYY-MM-DD)")
    parser.add_argument(
        "--stages",
        nargs="*",
        default=None,
        help=f"subset of stages to run (default: all built). Planned: {PLANNED_STAGES}",
    )
    args = parser.parse_args()

    config = load_config()
    requested = args.stages if args.stages is not None else list(STAGES)

    missing = [s for s in requested if s not in STAGES]
    if missing:
        built = list(STAGES) or "none yet (Phase 0 scaffold)"
        print(f"ERROR: stage(s) not built yet: {missing}. Built stages: {built}", file=sys.stderr)
        return 2

    if not requested:
        print("Scaffold OK: config loads, no pipeline stages built yet (Phase 0).")
        print(f"Planned stages: {' -> '.join(PLANNED_STAGES)}")
        return 0

    for name in requested:
        print(f"=== stage: {name} ===")
        STAGES[name](config, args)  # type: ignore[operator]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
