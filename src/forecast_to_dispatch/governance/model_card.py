"""Model card generator: the model's passport, built from real run artifacts.

Why this matters: a model card written by hand goes stale the day after the
model retrains. This one is *generated* from the registry, the training
metrics JSON, and the backtest results — so the card a reviewer reads is, by
construction, the card of the model actually serving. Completeness is a
release gate: governance.check fails if any required section is missing.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REQUIRED_SECTIONS = [
    "## Intended use",
    "## Training data",
    "## Evaluation protocol",
    "## Performance",
    "## Tail behavior",
    "## Limitations",
    "## Governance controls",
    "## Ownership",
]


def generate_model_card(
    config: dict[str, Any],
    models_root: Path,
    backtest_results: dict[str, Any] | None,
    out_path: Path,
    registry_path: Path | None = None,
) -> Path:
    registry_path = registry_path or models_root / "registry.json"
    registry = json.loads(registry_path.read_text())
    current = registry["current"]
    metrics_all = json.loads((models_root / current / "metrics.json").read_text())
    m = metrics_all["models"]["lgbm"]
    cl = metrics_all["models"]["climatology"]
    ls = metrics_all["models"].get("lstm", {})
    b = config["battery"]

    bt = ""
    if backtest_results:
        s = backtest_results
        bt = (
            f"Walk-forward backtest ({s['n_days']} out-of-sample days, "
            f"{s['window'][0]} → {s['window'][1]}): settled revenue "
            f"${s['totals']['governed']:,.0f} = **{s['revenue_capture_pct']:.1f}% revenue "
            f"capture** vs the perfect-foresight ceiling; +{s['uplift_vs_naive_pct']:.0f}% vs "
            f"the naive floor. Ancillary services carried "
            f"{100 * sum(v for k, v in s['governed_streams'].items() if k not in ('energy', 'degradation')) / max(sum(v for k, v in s['governed_streams'].items() if k != 'degradation'), 1e-9):.0f}% "
            "of gross revenue."
        )

    card = f"""# Model card — ERCOT price-quantile forecaster (production)

*Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} from `models/registry.json`
and the registered run artifacts. Do not edit by hand — regenerate via
`python -m forecast_to_dispatch.governance.check --refresh-artifacts`.*

**Model version:** `{current}` (challenger: `{registry.get('challenger', 'n/a')}`,
previous: `{registry.get('previous') or 'none'}`)

## Intended use

Day-ahead probabilistic forecasting of hourly ERCOT real-time settlement point
prices at {config['market']['hub']}, issued 09:00 D-1, to drive co-optimized
energy + ancillary-service dispatch of a {b['power_mw']:.0f} MW /
{b['energy_mwh']:.0f} MWh battery. **Not** for financial trading advice, other
hubs/ISOs without revalidation, or intraday re-forecasting.

## Training data

Public ERCOT archives via `gridstatus` (no paid/proprietary data): DAM and RTM
settlement point prices, DAM ancillary clearing prices (Reg Up/Down, RRS,
Non-Spin, ECRS), hourly system load, and wind/solar generation.
Training window: {metrics_all['train_window'][0][:10]} → {metrics_all['train_window'][1][:10]}
({metrics_all['n_train']:,} hourly observations, 28 leakage-safe features with
declared minimum legal lags).

## Evaluation protocol

Chronological split (test = strictly future, {metrics_all['n_test']:,} hours);
pinball loss (proper score), central-interval coverage, and scarcity recall
(P95 vs the training-window P99 threshold of
${m['scarcity_threshold']:,.0f}/MWh). Intervals are width-scaled conformally
calibrated on the last quarter of the training window.

## Performance

| Metric | This model (LGBM+CQR) | Climatology | LSTM challenger |
|---|---|---|---|
| Pinball (mean) | **{m['pinball_mean']:.2f}** | {cl['pinball_mean']:.2f} | {ls.get('pinball_mean', float('nan')):.2f} |
| 90% interval coverage | **{m['coverage_90']:.2f}** | {cl['coverage_90']:.2f} | {ls.get('coverage_90', float('nan')):.2f} |
| P50 MAE ($/MWh) | **{m['mae_p50']:.1f}** | {cl['mae_p50']:.1f} | — |

{bt}

## Tail behavior

Scarcity recall {m['scarcity_recall']:.2f} on {m['scarcity_hours']} realized
scarcity hours: the P95 flagged {m['scarcity_recall']:.0%} of hours that
settled above the scarcity threshold. The model reliably calls the *timing*
of stressed evenings but under-calls extreme *magnitudes* (see fig05, fig10);
dispatch positioning is therefore driven by the full quantile fan, not P50.

## Limitations

- **Exogenous drivers are persistence proxies** (lagged actual load/wind/solar):
  ERCOT's historical day-ahead forecast archives expire and cannot be
  reconstructed. Production would consume live forecast feeds — reported
  results are a conservative floor.
- **Single-season study** (Jan–Jun 2024, one hub). Summer regimes and other
  hubs require revalidation; the drift monitor is the tripwire.
- **AS prices forecast by persistence** (prior-day clearing prices), and
  reserves settle on capacity without deployment modeling.
- **Price-taker assumption**: valid for 100 MW at a liquid hub, not for a
  fleet.
- LSTM challenger underperforms at this data volume; retained in the registry
  for future comparison, not serving.

## Governance controls

Leakage truncation proof, conformal coverage gate (|coverage − 0.90| ≤
{config['forecast']['coverage_tolerance']}), PSI/KS drift monitor
(thresholds {config['governance']['drift']['psi_threshold']} /
{config['governance']['drift']['ks_pvalue_threshold']}), append-only
hash-chained audit log, human-in-the-loop approval bound to schedule hashes,
versioned registry with one-command rollback. All enforced by
`python -m forecast_to_dispatch.governance.check` (CI release gate).

## Ownership

Model owner: forecasting lead (this repository's maintainer). Dispatch owner:
optimization lead. Operations owner: on-call operator with HITL approval
authority. Escalation: any gate failure blocks release; rollback playbook in
`governance/rollback.py`.
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(card, encoding="utf-8")
    return out_path


def check_completeness(card_path: Path) -> tuple[bool, str]:
    """The gate: every required section present and non-trivial."""
    import re

    if not card_path.exists():
        return False, f"model card missing: {card_path}"
    text = card_path.read_text(encoding="utf-8")
    missing = [s for s in REQUIRED_SECTIONS if s not in text]
    if missing:
        return False, f"model card missing sections: {missing}"
    # Standalone 'nan' tokens = unfilled metrics ('Governance' contains 'nan';
    # a word-boundary match avoids that false alarm).
    if re.search(r"(?<![A-Za-z])nan(?![A-Za-z])", text, re.IGNORECASE):
        return False, "model card contains NaN metrics — regenerate from a real run"
    return True, "complete"
