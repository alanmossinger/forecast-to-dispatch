"""Governance figures (fig11, fig12, fig24): oversight made visible.

Why this matters: for the AI-governance reader these three figures ARE the
product — drift detection with thresholds (monitoring), a pass/fail gate
matrix (enforcement), and a real audit-trail timeline (accountability). Each
is generated from the live artifacts, never mocked.
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from forecast_to_dispatch.viz.style import PALETTE, apply_style, save_fig, titled

FEATURE_LEGEND = {
    "rtm_lag48": "RT price (2d lag)",
    "rtm_roll24_std": "RT volatility",
    "dam_lag24": "DA price (1d lag)",
    "net_load_lag48": "Net load (2d lag)",
    "prediction_q50": "Model P50 output",
}


def fig11_drift_monitor(
    rolling: pd.DataFrame, psi_threshold: float, save: bool = True
) -> plt.Figure:
    """Rolling PSI vs the training reference, with the action threshold."""
    apply_style()
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    idx = rolling.index.tz_localize(None) if rolling.index.tz is not None else rolling.index
    for col in rolling.columns:
        ax.plot(idx, rolling[col], lw=2, marker="o", ms=4, label=FEATURE_LEGEND.get(col, col))
    ax.axhline(
        psi_threshold,
        color=PALETTE["vermillion"],
        lw=1.6,
        ls="--",
        label=f"action threshold (PSI {psi_threshold})",
    )
    ax.axhspan(0.1, psi_threshold, color=PALETTE["yellow"], alpha=0.15)
    ax.text(idx[0], 0.11, "watch zone (0.1–0.2)", fontsize=8.5, color="#777777", va="bottom")
    ax.set_ylabel("PSI vs training reference")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(ncol=3, fontsize=9, loc="upper left")
    peak = rolling.max().max()
    verdict = (
        "regime shift would trip the gate before it costs money"
        if peak < psi_threshold
        else "threshold crossings visible — exactly what the gate exists to catch"
    )
    titled(
        ax,
        f"Drift is watched weekly, not assumed away: {verdict}",
        "PSI of monitored features and the model's own P50 output, weekly windows vs the "
        "training reference",
    )
    if save:
        save_fig(fig, "fig11_drift_monitor")
    return fig


def fig12_gate_status(results: list[dict], save: bool = True) -> plt.Figure:
    """The release-gate matrix: an ungoverned model literally cannot ship."""
    apply_style()
    fig, ax = plt.subplots(figsize=(11.5, 0.62 * len(results) + 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, len(results) - 0.5)
    ax.axis("off")
    for i, r in enumerate(reversed(results)):
        color = PALETTE["green"] if r["passed"] else PALETTE["vermillion"]
        mark = "PASS" if r["passed"] else "FAIL"
        ax.add_patch(plt.Rectangle((0, i - 0.36), 1.25, 0.72, color=color, alpha=0.9))
        ax.text(
            0.62, i, mark, ha="center", va="center", color="white", fontweight="bold", fontsize=11
        )
        ax.text(1.5, i, f"{r['gate']}  {r['name']}", va="center", fontsize=11.5, fontweight="bold")
        detail = r["detail"] if len(r["detail"]) < 92 else r["detail"][:89] + "..."
        ax.text(1.5, i - 0.27, detail, va="center", fontsize=8.5, color="#555555")
    n_pass = sum(r["passed"] for r in results)
    titled(
        ax,
        f"Release gates: {n_pass}/{len(results)} passing — CI blocks the merge on any failure",
        "python -m forecast_to_dispatch.governance.check — the single command CI runs; "
        "exit code enforces the verdict",
    )
    if save:
        save_fig(fig, "fig12_gate_status")
    return fig


def fig24_audit_trail_flow(records: list[dict], save: bool = True) -> plt.Figure:
    """One delivery day's real audit records as a decision timeline."""
    apply_style()
    events = records[-8:]  # the most recent decision sequence
    colors = {
        "forecast_issued": PALETTE["orange"],
        "schedule_created": PALETTE["blue"],
        "hitl_approval": PALETTE["green"],
        "dispatch_settled": PALETTE["purple"],
        "model_rollback": PALETTE["vermillion"],
    }
    fig, ax = plt.subplots(figsize=(12.5, 1.1 * len(events) + 2.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.6, len(events) - 0.1)
    ax.axis("off")
    for i, rec in enumerate(reversed(events)):
        y = i
        c = colors.get(rec["event_type"], "#888888")
        ax.add_patch(plt.Circle((0.55, y), 0.16, color=c, zorder=3))
        if i > 0:
            ax.plot([0.55, 0.55], [y - 1 + 0.2, y - 0.2], color="#BBBBBB", lw=1.4, zorder=1)
        payload_bits = []
        pl = rec["payload"]
        for key in (
            "day",
            "model_version",
            "approver",
            "expected_profit",
            "settled_revenue",
            "schedule_id",
            "reason",
            "input_hash",
        ):
            if key in pl:
                val = pl[key]
                if isinstance(val, float):
                    val = f"${val:,.0f}"
                if isinstance(val, str) and len(val) > 16 and key.endswith(("hash", "_id")):
                    val = val[:12] + "…"
                payload_bits.append(f"{key}={val}")
        ax.text(
            1.0,
            y + 0.14,
            rec["event_type"].replace("_", " ").upper(),
            fontsize=11,
            fontweight="bold",
            va="center",
            color=c,
        )
        ax.text(
            1.0,
            y - 0.20,
            f"{rec['ts_utc'][:19]}Z · actor={rec['actor']} · " + " · ".join(payload_bits),
            fontsize=8.6,
            va="center",
            color="#555555",
        )
        ax.text(
            9.9,
            y - 0.20,
            f"sha {rec['hash'][:10]}…",
            fontsize=8,
            va="center",
            ha="right",
            color="#999999",
            family="monospace",
        )
    titled(
        ax,
        "Every decision is a chained, tamper-evident record: forecast → schedule → human "
        "approval → settlement",
        "Excerpt of the real append-only audit log (JSONL, SHA-256 hash chain) — editing any "
        "historical record breaks verification",
    )
    if save:
        save_fig(fig, "fig24_audit_trail_flow")
    return fig
