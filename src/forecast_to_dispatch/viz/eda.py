"""Market-EDA figures (fig15-fig18): the business case for tail forecasting.

Why this matters: before any model is trusted, these four figures establish the
market problem in the executive's language — revenue lives in a small number of
scarcity hours, concentrated in a predictable evening window, on episodic days.
Every figure computes its headline number from the data and states it in the
title (house style: title = takeaway, subtitle = detail — see viz/style.py).
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecast_to_dispatch.viz.style import PALETTE, apply_style, save_fig, titled


def fig15_price_duration_curve(panel: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Price duration curve with the scarcity tail shaded.

    Why this matters: shows what share of the total price mass sits in the top
    1% of hours — the quantified business case for forecasting the tails.
    """
    apply_style()
    prices = panel["rtm_price"].sort_values(ascending=False).to_numpy()
    pct_hours = 100 * np.arange(1, len(prices) + 1) / len(prices)
    top1_share = prices[: max(1, len(prices) // 100)].sum() / prices.sum()

    fig, ax = plt.subplots()
    ax.plot(pct_hours, prices, color=PALETTE["blue"], lw=2)
    ax.fill_between(pct_hours, prices, where=pct_hours <= 1, color=PALETTE["vermillion"], alpha=0.5)
    ax.set_yscale("symlog", linthresh=100)
    ax.set_xlabel("% of hours (sorted from most to least expensive)")
    ax.set_ylabel("RTM price ($/MWh, symlog)")
    titled(
        ax,
        f"The top 1% of hours carries {top1_share:.0%} of all price mass — "
        "revenue lives in the tail",
        "ERCOT HB_HOUSTON real-time price duration curve; scarcity tail (top 1% of hours) shaded",
    )
    if save:
        save_fig(fig, "fig15_price_duration_curve")
    return fig


def fig16_hourly_price_heatmap(panel: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Mean RTM price by hour-of-day x month: the scarcity window made visible.

    Why this matters: the battery must be fully charged before the hours this
    map lights up — the evening net-load peak is the revenue window the whole
    system is built to hit.
    """
    apply_style()
    df = panel.copy()
    grid = df.pivot_table(
        values="rtm_price", index=df.index.hour, columns=df.index.month, aggfunc="mean"
    )
    peak_hour = grid.mean(axis=1).idxmax()

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(grid, aspect="auto", cmap="inferno", origin="lower")
    ax.set_xticks(range(grid.shape[1]), [f"{m:02d}" for m in grid.columns])
    ax.set_yticks(range(0, 24, 2), range(0, 24, 2))
    ax.set_xlabel("Month (2024)")
    ax.set_ylabel("Hour of day (local)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Mean RTM price ($/MWh)")
    titled(
        ax,
        f"Scarcity clusters in the evening — hour {peak_hour}:00 is the priciest on average",
        "Mean ERCOT HB_HOUSTON real-time price by hour of day and month",
    )
    if save:
        save_fig(fig, "fig16_hourly_price_heatmap")
    return fig


def fig17_dam_rtm_spread_dist(panel: pd.DataFrame, save: bool = True) -> plt.Figure:
    """DART spread distribution + duration curve: the arbitrage product itself.

    Why this matters: the width and skew of the DAM-RTM spread set the revenue
    ceiling for any storage arbitrage strategy; a battery earns nothing from a
    flat spread no matter how good the forecast is.
    """
    apply_style()
    spread = panel["dart_spread"]
    neg_share = (spread < 0).mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    clipped = spread.clip(-100, 100)
    ax1.hist(clipped, bins=80, color=PALETTE["sky"], edgecolor="none")
    ax1.axvline(0, color=PALETTE["black"], lw=1)
    ax1.axvline(
        spread.mean(),
        color=PALETTE["vermillion"],
        lw=1.5,
        ls="--",
        label=f"mean {spread.mean():+.1f} $/MWh",
    )
    ax1.set_xlabel("DART spread = DAM − RTM ($/MWh, clipped ±100)")
    ax1.set_ylabel("Hours")
    ax1.legend()

    sorted_spread = spread.sort_values(ascending=False).to_numpy()
    pct = 100 * np.arange(1, len(sorted_spread) + 1) / len(sorted_spread)
    ax2.plot(pct, sorted_spread, color=PALETTE["orange"], lw=2)
    ax2.axhline(0, color=PALETTE["black"], lw=1)
    ax2.set_yscale("symlog", linthresh=50)
    ax2.set_xlabel("% of hours (sorted by spread)")
    ax2.set_ylabel("DART spread ($/MWh, symlog)")

    titled(
        ax1,
        f"The spread is fat-tailed both ways — RTM exceeds DAM in {neg_share:.0%} of hours",
        "Distribution (left) and duration curve (right) of the day-ahead-minus-real-time spread",
    )
    if save:
        save_fig(fig, "fig17_dam_rtm_spread_dist")
    return fig


def fig19_feature_target_relationships(
    panel: pd.DataFrame,
    exog: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    save: bool = True,
) -> plt.Figure:
    """Small multiples: market physics (top) vs what the model may see (bottom).

    Why this matters: before trusting any ML, we show the physical logic the
    model is supposed to learn — prices rise with net load and fall with
    renewables (top row, same-hour physics). The bottom row shows the SAME
    relationships through the only lens the model is allowed: leakage-safe
    lagged features vs the next-day target. The signal survives the lag —
    weaker, but real. That gap between the rows is, precisely, the forecasting
    problem.
    """
    apply_style()
    joined = panel.join(exog, how="inner")

    def _scatter(ax, x, yy, xlabel, n_bins=20):
        ax.scatter(x, yy, s=4, alpha=0.15, color=PALETTE["blue"], rasterized=True)
        bins = pd.qcut(x, n_bins, duplicates="drop")
        binned = yy.groupby(bins, observed=True).mean()
        centers = [iv.mid for iv in binned.index]
        ax.plot(centers, binned, color=PALETTE["vermillion"], lw=2.5, label="binned mean")
        ax.set_yscale("symlog", linthresh=100)
        ax.set_xlabel(xlabel)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), sharey=True)

    _scatter(
        axes[0, 0], joined["net_load_mw"] / 1e3, joined["rtm_price"], "Net load (GW), same hour"
    )
    _scatter(
        axes[0, 1], joined["solar_mw"] / 1e3, joined["rtm_price"], "Solar output (GW), same hour"
    )
    _scatter(
        axes[0, 2], joined["load_mw"] / 1e3, joined["rtm_price"], "System load (GW), same hour"
    )
    axes[0, 0].set_ylabel("RTM price ($/MWh, symlog)")

    common = X.index.intersection(y.index)
    _scatter(
        axes[1, 0],
        X.loc[common, "net_load_lag48"] / 1e3,
        y.loc[common],
        "Net load 48h earlier (GW)",
    )
    _scatter(
        axes[1, 1], X.loc[common, "rtm_lag168"], y.loc[common], "RTM price 1 week earlier ($/MWh)"
    )
    _scatter(axes[1, 2], X.loc[common, "dam_lag24"], y.loc[common], "DAM price, prior day ($/MWh)")
    axes[1, 0].set_ylabel("Next-day RTM price ($/MWh)")

    for ax in axes.flat:
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Prices follow net load — and the signal survives the leakage-safe lag",
        x=0.01,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    axes[0, 0].set_title(
        "Market physics (contemporaneous)", loc="left", fontsize=11, fontweight="normal"
    )
    axes[1, 0].set_title(
        "The model's causal view (lagged features only)",
        loc="left",
        fontsize=11,
        fontweight="normal",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save:
        save_fig(fig, "fig19_feature_target_relationships")
    return fig


def fig18_scarcity_calendar(
    panel: pd.DataFrame, scarcity_percentile: float = 0.99, save: bool = True
) -> plt.Figure:
    """Daily maximum price strip with scarcity days flagged.

    Why this matters: scarcity is episodic — it arrives in bursts, not on a
    schedule. A model (and its drift monitor) must handle regime bursts, and an
    operator must know how many 'paydays' a season actually contains.
    """
    apply_style()
    threshold = panel["rtm_price"].quantile(scarcity_percentile)
    daily_max = panel["rtm_price"].resample("1D").max()
    scarce = daily_max >= threshold

    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.bar(
        daily_max.index,
        daily_max,
        width=0.9,
        color=[PALETTE["vermillion"] if s else PALETTE["blue"] for s in scarce],
    )
    ax.axhline(
        threshold,
        color=PALETTE["black"],
        lw=1.2,
        ls="--",
        label=f"P{scarcity_percentile * 100:.0f} scarcity threshold (${threshold:,.0f})",
    )
    ax.set_yscale("symlog", linthresh=200)
    ax.set_ylabel("Daily max RTM price ($/MWh, symlog)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.legend()
    titled(
        ax,
        f"Scarcity is episodic: {int(scarce.sum())} spike days out of {len(daily_max)}, "
        "arriving in bursts",
        "Daily maximum ERCOT HB_HOUSTON real-time price; scarcity days (above threshold) in red",
    )
    if save:
        save_fig(fig, "fig18_scarcity_calendar")
    return fig
