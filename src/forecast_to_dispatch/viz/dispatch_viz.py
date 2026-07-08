"""Dispatch figure (fig06): one day of battery decisions, readable by anyone.

Why this matters: this is where the abstract forecast becomes concrete
operator behavior — buy the cheap hours, hold charge for the evening window,
sell energy into the spike, and rent out standby capacity in between. If a
reviewer can't see buy-low/sell-high/hold-for-scarcity in this picture, the
optimizer isn't doing its job.
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from forecast_to_dispatch.optimize.dispatch import ALL_PRODUCTS, BatterySpec
from forecast_to_dispatch.viz.style import PALETTE, apply_style, save_fig

PRODUCT_COLORS = {
    "regup": PALETTE["purple"],
    "regdn": "#B5A2C8",
    "rrs": PALETTE["sky"],
    "nspin": PALETTE["yellow"],
    "ecrs": "#7A9E7E",
}


def fig06_dispatch_day(
    schedule: pd.DataFrame,
    prices: pd.DataFrame,
    spec: BatterySpec,
    realized_price: pd.Series | None = None,
    save: bool = True,
) -> plt.Figure:
    """Three aligned panels: the price signal, the power decisions, the SOC."""
    apply_style()
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(12.5, 10.5), sharex=True, gridspec_kw={"height_ratios": [1.1, 1.2, 0.9]}
    )
    # Plot in wall-clock local time; matplotlib renders tz-aware stamps in UTC.
    orig_idx = schedule.index
    idx = orig_idx.tz_localize(None)
    if realized_price is not None:
        realized_price = realized_price.loc[orig_idx].set_axis(idx)
    prices = prices.set_axis(idx)
    schedule = schedule.set_axis(idx)

    # Panel 1 — the price signal the optimizer saw (and what actually happened)
    ax1.plot(
        idx,
        prices["energy_price"],
        color=PALETTE["orange"],
        lw=2,
        label="forecast energy price (dispatch signal)",
    )
    if realized_price is not None:
        ax1.plot(
            idx,
            realized_price,
            color=PALETTE["blue"],
            lw=1.6,
            label="realized RT price (settlement)",
        )
    ax1.set_ylabel("$/MWh")
    ax1.set_yscale("symlog", linthresh=150)
    ax1.legend(loc="upper left", fontsize=9)

    # Panel 2 — power decisions: discharge above zero, charge below, reserves stacked
    ax2.bar(
        idx,
        schedule["discharge_mw"],
        width=0.036,
        color=PALETTE["green"],
        label="discharge (sell energy)",
    )
    ax2.bar(
        idx, -schedule["charge_mw"], width=0.036, color=PALETTE["blue"], label="charge (buy energy)"
    )
    bottom = schedule["discharge_mw"].copy()
    for k in ALL_PRODUCTS:
        vals = schedule[f"r_{k}_mw"]
        if vals.sum() < 1e-6:
            continue
        ax2.bar(
            idx,
            vals,
            width=0.036,
            bottom=bottom,
            color=PRODUCT_COLORS[k],
            alpha=0.75,
            label=f"{k.upper()} reserve",
        )
        bottom = bottom + vals
    ax2.axhline(0, color=PALETTE["black"], lw=0.8)
    ax2.axhline(spec.power_mw, color=PALETTE["black"], lw=0.8, ls=":")
    ax2.axhline(-spec.power_mw, color=PALETTE["black"], lw=0.8, ls=":")
    ax2.set_ylabel("MW (± = sell/buy)")
    # Legend lives at the figure bottom — the stacked bars fill the panel.
    handles, labels_ = ax2.get_legend_handles_labels()
    fig.legend(
        handles,
        labels_,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=4,
        fontsize=9,
        frameon=False,
    )

    # Panel 3 — state of charge
    ax3.fill_between(idx, schedule["soc_mwh"], color=PALETTE["vermillion"], alpha=0.25)
    ax3.plot(idx, schedule["soc_mwh"], color=PALETTE["vermillion"], lw=2, label="state of charge")
    ax3.axhline(
        spec.soc_max_mwh,
        color=PALETTE["black"],
        lw=0.8,
        ls=":",
        label=f"capacity ({spec.soc_max_mwh:.0f} MWh)",
    )
    ax3.set_ylabel("MWh")
    ax3.set_ylim(0, spec.soc_max_mwh * 1.08)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:00"))
    ax3.set_xlabel(f"Hour of {idx[0]:%A, %B %d %Y} (local)")
    ax3.legend(loc="upper left", fontsize=9)

    profit = schedule["expected_profit"].sum()
    fig.suptitle(
        "Buy the trough, hold for the evening, sell the spike — and rent out standby capacity",
        x=0.01,
        y=0.995,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.962,
        f"Co-optimized 100 MW / 200 MWh dispatch on day-ahead forecast prices; "
        f"expected profit at forecast prices ${profit:,.0f}",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.945))
    if save:
        save_fig(fig, "fig06_dispatch_day")
    return fig
