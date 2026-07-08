"""Shared plot theme: one visual identity for every figure in the repo.

Why this matters: the figures ARE the deliverable for the executive audience.
A consistent, colorblind-safe style with conclusion-as-title makes the whole
report read as one system, and guarantees any reviewer — including the ~5% of
men with color-vision deficiency — can read the money charts correctly.

Conventions enforced here (per the style guide §6):
- Okabe-Ito colorblind-safe palette, with fixed semantic roles
  (price=blue, forecast band=orange, revenue=green, alerts=vermillion).
- Title states the takeaway; subtitle carries the axis-level detail.
- Every figure saved to reports/figures/ as PNG at 150+ DPI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from forecast_to_dispatch.config import REPO_ROOT

FIGURES_DIR = REPO_ROOT / "reports" / "figures"

# Okabe-Ito palette — the standard colorblind-safe set.
PALETTE = {
    "blue": "#0072B2",  # realized / actual price
    "orange": "#E69F00",  # forecasts and quantile bands
    "green": "#009E73",  # revenue, positive outcomes
    "vermillion": "#D55E00",  # alerts, scarcity, failures
    "purple": "#CC79A7",  # ancillary services
    "sky": "#56B4E9",  # secondary series (e.g. DAM vs RTM)
    "yellow": "#F0E442",  # highlights
    "black": "#000000",
}

# Fixed semantic order for categorical plots (policies, products).
CYCLE = [
    PALETTE["blue"],
    PALETTE["orange"],
    PALETTE["green"],
    PALETTE["vermillion"],
    PALETTE["purple"],
    PALETTE["sky"],
    PALETTE["yellow"],
]


def apply_style() -> None:
    """Apply the repo-wide matplotlib theme. Call once per notebook/script."""
    mpl.rcParams.update(
        {
            "figure.figsize": (11, 5.5),
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "axes.prop_cycle": mpl.cycler(color=CYCLE),
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.family": "sans-serif",
            "font.size": 11,
            "legend.frameon": False,
            "legend.fontsize": 10,
        }
    )


def titled(ax: plt.Axes, takeaway: str, detail: str) -> plt.Axes:
    """Set a conclusion-as-title plus a smaller detail subtitle.

    Why this matters: an executive skims titles only. 'Battery holds charge
    for the evening scarcity window' communicates; 'SOC vs time' does not.
    """
    ax.set_title(takeaway, pad=22)
    ax.text(
        0,
        1.02,
        detail,
        transform=ax.transAxes,
        fontsize=10,
        color="#555555",
        va="bottom",
    )
    return ax


def save_fig(fig: plt.Figure, name: str) -> Path:
    """Save a figure to reports/figures/<name>.png at publication DPI.

    Fails loudly if the name doesn't follow the figNN_slug catalog convention,
    so the figure catalog in the style guide stays authoritative.
    """
    if not name.startswith("fig"):
        raise ValueError(f"Figure name must follow the catalog convention 'figNN_slug': {name}")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"{name}.png"
    fig.savefig(out)
    return out
