"""fig13: the system architecture — where human oversight and rollback sit.

Why this matters: this is the one-page answer to "how is an autonomous
real-money agent kept on a leash?" The decision pipeline runs left to right;
the governance rail runs underneath with explicit hooks into it; everything
funnels into a single CI-enforced release gate.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from forecast_to_dispatch.viz.style import PALETTE, apply_style, save_fig


def _box(ax, xy, w, h, title, sub, color, text_color="white", fontsize=10):
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.06",
            linewidth=0,
            facecolor=color,
            alpha=0.95,
        )
    )
    ax.text(
        x + w / 2,
        y + h * 0.62,
        title,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=text_color,
    )
    ax.text(
        x + w / 2,
        y + h * 0.28,
        sub,
        ha="center",
        va="center",
        fontsize=fontsize - 2.4,
        color=text_color,
        alpha=0.92,
    )


def _arrow(ax, a, b, color="#555555", style="-|>", lw=1.8, connstyle="arc3,rad=0.0"):
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle=style,
            mutation_scale=14,
            lw=lw,
            color=color,
            connectionstyle=connstyle,
            zorder=1,
        )
    )


def fig13_architecture(save: bool = True) -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(13.5, 8))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 8)
    ax.axis("off")

    blue, orange, green = PALETTE["blue"], PALETTE["orange"], PALETTE["green"]
    verm, black = PALETTE["vermillion"], "#333333"

    # ---- decision pipeline (top lane) ----
    ax.text(
        0.15,
        7.55,
        "DECISION PIPELINE (runs daily, 09:00 D-1)",
        fontsize=10.5,
        fontweight="bold",
        color="#444444",
    )
    _box(
        ax,
        (0.15, 6.1),
        2.0,
        1.1,
        "ERCOT public\narchives",
        "prices · AS · load · wind/solar",
        "#666666",
    )
    _box(ax, (2.55, 6.1), 2.0, 1.1, "Ingest", "gap-validated hourly panel", blue)
    _box(ax, (4.95, 6.1), 2.0, 1.1, "Features", "28 leakage-proofed inputs", blue)
    _box(ax, (7.35, 6.1), 2.0, 1.1, "Quantile forecast", "LGBM + conformal (P5–P95)", orange)
    _box(ax, (9.75, 6.1), 2.0, 1.1, "Dispatch optimizer", "cvxpy: energy + 5 AS products", green)
    _box(ax, (11.95, 6.1), 1.4, 1.1, "Market\nsettlement", "realized prices", "#666666")
    for x in (2.15, 4.55, 6.95, 9.35, 11.75):
        _arrow(ax, (x, 6.65), (x + 0.42, 6.65), color=black)

    # SHAP + serving (middle lane)
    _box(
        ax, (7.35, 4.55), 2.0, 0.95, "SHAP explainability", "drivers in market language", "#8A6FA8"
    )
    _arrow(ax, (8.35, 6.08), (8.35, 5.52), color="#8A6FA8")
    _box(ax, (9.75, 4.55), 2.0, 0.95, "FastAPI serving", "/forecast · /dispatch", "#4A7B9D")
    _arrow(ax, (10.75, 6.08), (10.75, 5.52), color="#4A7B9D")

    # ---- governance rail (bottom lane) ----
    ax.add_patch(
        FancyBboxPatch(
            (0.15, 0.35),
            13.1,
            3.5,
            boxstyle="round,pad=0.08",
            linewidth=1.4,
            edgecolor=verm,
            facecolor=verm,
            alpha=0.06,
        )
    )
    ax.text(
        0.4,
        3.55,
        "GOVERNANCE RAIL (NIST AI RMF · EU AI Act alignment)",
        fontsize=10.5,
        fontweight="bold",
        color=verm,
    )

    _box(ax, (0.45, 2.25), 1.85, 1.0, "Model card", "generated from real\nrun metrics", verm)
    _box(ax, (2.55, 2.25), 1.85, 1.0, "Risk register", "10 risks, schema-\nvalidated YAML", verm)
    _box(ax, (4.65, 2.25), 1.85, 1.0, "Drift monitor", "PSI + KS vs training\nreference", verm)
    _box(ax, (6.75, 2.25), 1.85, 1.0, "Audit log", "append-only,\nSHA-256 chained", verm)
    _box(
        ax,
        (8.85, 2.25),
        1.85,
        1.0,
        "HITL approval",
        "human sign-off bound\nto schedule hash",
        black,
    )
    _box(ax, (10.95, 2.25), 1.85, 1.0, "Rollback", "one command to\nprevious model", verm)

    _box(
        ax,
        (3.6, 0.6),
        6.3,
        1.0,
        "governance.check — the single release gate",
        "9 gates · any failure = non-zero exit = CI blocks the merge",
        "#7A1F1F",
        fontsize=11,
    )
    for x in (1.35, 3.45, 5.55, 7.65, 9.75, 11.85):
        _arrow(
            ax,
            (x, 2.2),
            (min(max(x, 4.0), 9.6), 1.66),
            color=verm,
            lw=1.3,
            connstyle="arc3,rad=0.12" if x < 4 or x > 9.5 else "arc3,rad=0.0",
        )

    # hooks between lanes
    _arrow(ax, (5.95, 6.08), (5.55, 3.3), color=verm, lw=1.3, connstyle="arc3,rad=0.25")
    _arrow(ax, (8.1, 4.5), (7.65, 3.3), color=verm, lw=1.3, connstyle="arc3,rad=0.15")
    _arrow(ax, (9.8, 2.75), (10.55, 6.05), color=black, lw=1.6, connstyle="arc3,rad=-0.35")
    ax.text(
        9.15,
        4.2,
        "no approval →\nno deployment",
        fontsize=8.4,
        color=black,
        ha="center",
        fontstyle="italic",
    )
    _arrow(ax, (11.9, 3.28), (8.6, 6.05), color=verm, lw=1.3, connstyle="arc3,rad=0.3")
    ax.text(
        12.35, 4.1, "re-points\nregistry", fontsize=8.4, color=verm, ha="center", fontstyle="italic"
    )

    ax.set_title(
        "Forecast → dispatch → settlement, with governance wired in — not bolted on",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    ax.text(
        0,
        1.005,
        "Every rail component is code in this repository, exercised by tests "
        "and enforced by the CI release gate",
        transform=ax.transAxes,
        fontsize=10,
        color="#555555",
        va="bottom",
    )
    if save:
        save_fig(fig, "fig13_architecture")
    return fig
