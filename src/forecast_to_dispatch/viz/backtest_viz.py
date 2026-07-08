"""Backtest figures (fig07-fig10, fig21-fig23): the money story, honestly scaled.

Why this matters: fig07 is the single chart the whole repository exists to
produce — governed revenue between an honest floor and an unreachable ceiling.
The remaining figures answer the follow-up questions an asset owner asks next:
how much of the available value was captured, where the revenue came from,
what imperfect forecasts cost, and whether the behavior was systematic or luck.
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecast_to_dispatch.optimize.dispatch import ALL_PRODUCTS
from forecast_to_dispatch.viz.style import PALETTE, apply_style, save_fig, titled

POLICY_COLORS = {
    "perfect": "#999999",
    "governed": PALETTE["green"],
    "naive": PALETTE["sky"],
}
POLICY_LABELS = {
    "perfect": "perfect foresight (ceiling)",
    "governed": "governed forecast-driven",
    "naive": "naive average-day (floor)",
}
STREAM_LABELS = {
    "energy": "Energy arbitrage",
    "regup": "Reg Up",
    "regdn": "Reg Down",
    "rrs": "RRS",
    "nspin": "Non-Spin",
    "ecrs": "ECRS",
}
STREAM_COLORS = {
    "energy": PALETTE["blue"],
    "regup": PALETTE["purple"],
    "regdn": "#B5A2C8",
    "rrs": PALETTE["sky"],
    "nspin": PALETTE["yellow"],
    "ecrs": "#7A9E7E",
}


def fig07_cumulative_revenue(daily: pd.DataFrame, summary: dict, save: bool = True) -> plt.Figure:
    """THE headline chart: cumulative settled revenue, three policies."""
    apply_style()
    fig, ax = plt.subplots(figsize=(12.5, 6))
    idx = daily.index.tz_localize(None)
    for p in ("perfect", "governed", "naive"):
        cum = daily[f"{p}_revenue"].cumsum() / 1e6
        ax.plot(
            idx,
            cum,
            lw=2.4 if p == "governed" else 1.8,
            color=POLICY_COLORS[p],
            label=POLICY_LABELS[p],
        )
        ax.annotate(
            f"${cum.iloc[-1]:.2f}M",
            (idx[-1], cum.iloc[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            color=POLICY_COLORS[p],
            fontweight="bold",
            fontsize=10,
            va="center",
        )
    ax.set_ylabel("Cumulative settled revenue ($M)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(loc="upper left")
    ax.margins(x=0.09)
    capture = summary["revenue_capture_pct"]
    uplift = summary["uplift_vs_naive_pct"]
    titled(
        ax,
        f"The governed agent captures {capture:.0f}% of the theoretical ceiling "
        f"({uplift:+.0f}% vs naive)",
        f"Cumulative revenue over {summary['n_days']} out-of-sample days — dispatched on "
        "forecasts, settled at realized prices, 100 MW / 200 MWh",
    )
    if save:
        save_fig(fig, "fig07_cumulative_revenue")
    return fig


def fig08_revenue_capture_bar(summary: dict, save: bool = True) -> plt.Figure:
    """Revenue capture: how much of the available value each policy realized."""
    apply_style()
    totals = summary["totals"]
    pct = {p: 100 * totals[p] / totals["perfect"] for p in ("naive", "governed", "perfect")}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    order = ["naive", "governed", "perfect"]
    bars = ax.bar(
        [POLICY_LABELS[p] for p in order],
        [pct[p] for p in order],
        color=[POLICY_COLORS[p] for p in order],
        width=0.55,
    )
    for bar, p in zip(bars, order):
        ax.annotate(
            f"{pct[p]:.0f}%\n(${totals[p] / 1e6:.2f}M)",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_ylabel("% of perfect-foresight revenue")
    ax.set_ylim(0, 118)
    titled(
        ax,
        f"{pct['governed']:.0f}% of the theoretically available value, captured with "
        "honest forecasts",
        "Settled revenue as a share of the perfect-foresight ceiling, same battery and physics",
    )
    if save:
        save_fig(fig, "fig08_revenue_capture_bar")
    return fig


def fig09_revenue_stack(summary: dict, save: bool = True) -> plt.Figure:
    """Where governed revenue came from: energy vs each ancillary product."""
    apply_style()
    streams = summary["governed_streams"]
    order = ["energy", "ecrs", "nspin", "regdn", "regup", "rrs"]
    vals = [streams[s] / 1e6 for s in order]
    deg = streams["degradation"] / 1e6
    as_share = sum(streams[k] for k in ALL_PRODUCTS) / sum(
        streams[k] for k in ["energy"] + ALL_PRODUCTS
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(
        [STREAM_LABELS[s] for s in order], vals, color=[STREAM_COLORS[s] for s in order], width=0.6
    )
    ax.bar(["(degradation)"], [deg], color=PALETTE["vermillion"], width=0.6)
    for bar, v in zip(list(bars) + list(ax.containers[-1]), vals + [deg]):
        ax.annotate(
            f"${v:.2f}M",
            (bar.get_x() + bar.get_width() / 2, max(v, 0)),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.axhline(0, color=PALETTE["black"], lw=0.8)
    ax.set_ylabel("Settled revenue ($M)")
    titled(
        ax,
        f"Revenue stacking works: ancillary services carry {as_share:.0%} of gross revenue",
        "Governed policy revenue by stream, settled at realized prices (RTC+B co-optimization)",
    )
    if save:
        save_fig(fig, "fig09_revenue_stack")
    return fig


def fig10_cost_of_imperfection(daily: pd.DataFrame, save: bool = True) -> plt.Figure:
    """The governed-vs-perfect gap, tied to forecast error day by day."""
    apply_style()
    gap = (daily["perfect_revenue"] - daily["governed_revenue"]) / 1e3
    mae = daily["forecast_mae"]
    rho = float(np.corrcoef(mae, gap)[0, 1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    scarce = daily["realized_max_price"] >= daily["realized_max_price"].quantile(0.85)
    ax1.scatter(mae[~scarce], gap[~scarce], s=42, color=PALETTE["blue"], label="calm days")
    ax1.scatter(
        mae[scarce], gap[scarce], s=52, color=PALETTE["vermillion"], label="spike days (top 15%)"
    )
    coef = np.polyfit(mae, gap, 1)
    xs = np.linspace(mae.min(), mae.max(), 50)
    ax1.plot(
        xs,
        np.polyval(coef, xs),
        color=PALETTE["black"],
        lw=1.2,
        ls="--",
        label=f"trend (r = {rho:.2f})",
    )
    ax1.set_xlabel("Daily forecast MAE ($/MWh)")
    ax1.set_ylabel("Revenue left on the table ($k/day)")
    ax1.legend(fontsize=9)

    idx = daily.index.tz_localize(None)
    ax2.fill_between(idx, gap.cumsum(), color=PALETTE["vermillion"], alpha=0.35)
    ax2.plot(idx, gap.cumsum(), color=PALETTE["vermillion"], lw=2)
    ax2.set_ylabel("Cumulative gap to perfect foresight ($k)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    titled(
        ax1,
        "The cost of imperfect foresight is real, measured, and driven by spike days",
        "Daily gap to the perfect-foresight ceiling vs forecast error (left); cumulative gap "
        "(right). This gap is what leaky backtests hide.",
    )
    fig.tight_layout()
    if save:
        save_fig(fig, "fig10_cost_of_imperfection")
    return fig


def fig21_soc_heatmap(governed_schedule: pd.DataFrame, save: bool = True) -> plt.Figure:
    """SOC by hour x day across the whole backtest: systematic, not lucky."""
    apply_style()
    s = governed_schedule
    grid = s.pivot_table(
        values="soc_mwh", index=s.index.hour, columns=s.index.normalize(), aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
    ax.set_ylabel("Hour of day (local)")
    ax.set_xlabel("Backtest day")
    ticks = np.linspace(0, grid.shape[1] - 1, 8).astype(int)
    ax.set_xticks(ticks, [f"{grid.columns[i]:%b %d}" for i in ticks])
    ax.set_yticks(range(0, 24, 4), range(0, 24, 4))
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="State of charge (MWh)")
    evening_soc = grid.loc[17:20].mean().mean()
    titled(
        ax,
        f"The battery holds charge for the evening window every single day "
        f"(~{evening_soc:.0f} MWh entering it)",
        "Governed policy state of charge by hour and day, full backtest",
    )
    if save:
        save_fig(fig, "fig21_soc_heatmap")
    return fig


def fig22_reserve_allocation(governed_schedule: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Daily capacity allocation: how the battery's MW were rented out."""
    apply_style()
    s = governed_schedule
    day = s.index.normalize()
    daily = pd.DataFrame(
        {STREAM_LABELS[k]: s[f"r_{k}_mw"].groupby(day).sum() for k in ALL_PRODUCTS}
        | {"Energy discharge": s["discharge_mw"].groupby(day).sum()}
    )
    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    colors = [STREAM_COLORS[k] for k in ALL_PRODUCTS] + [PALETTE["green"]]
    ax.stackplot(
        daily.index.tz_localize(None),
        daily.T.to_numpy(),
        labels=daily.columns,
        colors=colors,
        alpha=0.85,
    )
    ax.set_ylabel("MW·h committed per day")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    titled(
        ax,
        "The same megawatts earn twice: reserves carry the quiet days, energy takes the spikes",
        "Governed policy daily capacity commitments by product (stacked), full backtest",
    )
    if save:
        save_fig(fig, "fig22_reserve_allocation")
    return fig


def fig23_monthly_revenue(daily: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Monthly settled revenue by policy: consistency, not one lucky spike."""
    apply_style()
    month = daily.index.to_period("M")
    monthly = daily.groupby(month)[["naive_revenue", "governed_revenue", "perfect_revenue"]].sum()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    xs = np.arange(len(monthly))
    for i, p in enumerate(("naive", "governed", "perfect")):
        ax.bar(
            xs + (i - 1) * 0.26,
            monthly[f"{p}_revenue"] / 1e6,
            0.26,
            color=POLICY_COLORS[p],
            label=POLICY_LABELS[p],
        )
    ax.set_xticks(xs, [str(m) for m in monthly.index])
    ax.set_ylabel("Settled revenue ($M)")
    ax.legend(fontsize=9)
    share = monthly["governed_revenue"] / monthly["perfect_revenue"]
    titled(
        ax,
        f"Capture is consistent across months ({share.min():.0%}–{share.max():.0%}), "
        "not one lucky spike",
        "Monthly settled revenue by policy, out-of-sample window",
    )
    if save:
        save_fig(fig, "fig23_monthly_revenue")
    return fig
