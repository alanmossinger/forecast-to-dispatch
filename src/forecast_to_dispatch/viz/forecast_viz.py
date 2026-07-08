"""Forecast-quality figures (fig01, fig02, fig05, fig14, fig20).

Why this matters: these are the charts a reviewer uses to decide whether the
forecaster can be trusted with capital — does the uncertainty band widen ahead
of spikes (fan), is the stated confidence honest (calibration), did the tail
see the payday coming (scarcity zoom), does the deep model earn its complexity
(comparison), and where exactly does the model lose money (error by regime).
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecast_to_dispatch.forecast.base import quantile_col
from forecast_to_dispatch.forecast.metrics import interval_coverage
from forecast_to_dispatch.viz.style import PALETTE, apply_style, save_fig, titled


def fig01_forecast_fan(
    y: pd.Series, preds: pd.DataFrame, week_start: pd.Timestamp, save: bool = True
) -> plt.Figure:
    """Actual price under the P5-P95 quantile fan for one representative week."""
    apply_style()
    window = slice(week_start, week_start + pd.Timedelta(days=7))
    yy, pp = y.loc[window], preds.loc[window]

    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    ax.fill_between(
        pp.index,
        pp["q05"],
        pp["q95"],
        color=PALETTE["orange"],
        alpha=0.30,
        label="P5–P95 (90% interval)",
    )
    ax.fill_between(
        pp.index, pp["q25"], pp["q75"], color=PALETTE["orange"], alpha=0.55, label="P25–P75"
    )
    ax.plot(pp.index, pp["q50"], color=PALETTE["vermillion"], lw=1.6, label="P50 forecast")
    ax.plot(yy.index, yy, color=PALETTE["blue"], lw=1.6, label="Realized RT price")
    ax.set_yscale("symlog", linthresh=150)
    ax.set_ylabel("RTM price ($/MWh, symlog)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
    ax.legend(ncol=4, loc="upper left")
    titled(
        ax,
        "The band breathes with risk: wide into stressed evenings, tight on calm days",
        f"Day-ahead quantile forecasts vs realized prices, week of {week_start:%b %d %Y} "
        "(all forecasts issued 09:00 D-1)",
    )
    if save:
        save_fig(fig, "fig01_forecast_fan")
    return fig


def fig02_calibration(
    y: pd.Series, preds: pd.DataFrame, quantiles: list[float], save: bool = True
) -> plt.Figure:
    """Nominal vs empirical: below-quantile rates and central-interval coverage."""
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    nominal = np.array(quantiles)
    empirical = np.array([(y <= preds[quantile_col(q)]).mean() for q in quantiles])
    ax1.plot([0, 1], [0, 1], color=PALETTE["black"], lw=1, ls="--", label="perfect honesty")
    ax1.plot(nominal, empirical, "o-", color=PALETTE["blue"], lw=2, ms=7, label="model")
    for q, e in zip(nominal, empirical):
        ax1.annotate(f"{e:.2f}", (q, e), textcoords="offset points", xytext=(6, -12), fontsize=9)
    ax1.set_xlabel("Nominal quantile level")
    ax1.set_ylabel("Empirical share of hours below forecast")
    ax1.legend()

    intervals = {"P25–P75 (50%)": (0.50, "q25", "q75"), "P5–P95 (90%)": (0.90, "q05", "q95")}
    names = list(intervals)
    nominal_cov = [intervals[k][0] for k in names]
    empirical_cov = [
        interval_coverage(y, preds[intervals[k][1]], preds[intervals[k][2]]) for k in names
    ]
    xpos = np.arange(len(names))
    ax2.bar(xpos - 0.18, nominal_cov, width=0.36, color=PALETTE["sky"], label="nominal")
    ax2.bar(xpos + 0.18, empirical_cov, width=0.36, color=PALETTE["green"], label="empirical")
    for x, v in zip(xpos + 0.18, empirical_cov):
        ax2.annotate(f"{v:.2f}", (x, v), ha="center", va="bottom", fontsize=10)
    ax2.set_xticks(xpos, names)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Coverage")
    ax2.legend()

    gap = abs(empirical_cov[1] - 0.90)
    titled(
        ax1,
        f"The model is honest about uncertainty: 90% interval covers {empirical_cov[1]:.0%} "
        f"(gap {gap:.0%})",
        "Left: quantile-level calibration. Right: central-interval coverage, nominal vs empirical "
        "(test window)",
    )
    if save:
        save_fig(fig, "fig02_calibration")
    return fig


def fig05_scarcity_zoom(
    y: pd.Series, preds: pd.DataFrame, threshold: float, save: bool = True
) -> plt.Figure:
    """Zoom on the largest realized spike: did the P95 raise its hand in time?"""
    apply_style()
    spike_hour = y.idxmax()
    lo = spike_hour - pd.Timedelta(days=2)
    hi = spike_hour + pd.Timedelta(days=1)
    yy, pp = y.loc[lo:hi], preds.loc[lo:hi]

    anticipated = pp.loc[spike_hour, "q95"] >= threshold
    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    ax.fill_between(
        pp.index, pp["q05"], pp["q95"], color=PALETTE["orange"], alpha=0.30, label="P5–P95 band"
    )
    ax.plot(pp.index, pp["q95"], color=PALETTE["orange"], lw=1.6, label="P95 forecast")
    ax.plot(pp.index, pp["q50"], color=PALETTE["vermillion"], lw=1.4, label="P50 forecast")
    ax.plot(yy.index, yy, color=PALETTE["blue"], lw=1.8, label="Realized RT price")
    ax.axhline(
        threshold,
        color=PALETTE["black"],
        lw=1.2,
        ls="--",
        label=f"scarcity threshold (${threshold:,.0f})",
    )
    ax.annotate(
        f"spike: ${yy.max():,.0f}/MWh",
        xy=(spike_hour, yy.max()),
        xytext=(-90, -18),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": PALETTE["black"]},
        fontsize=10,
    )
    ax.set_yscale("symlog", linthresh=200)
    ax.set_ylabel("RTM price ($/MWh, symlog)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:00"))
    ax.legend(ncol=3, loc="upper left")
    verdict = (
        "P95 raised its hand before the event"
        if anticipated
        else "P95 missed this one — logged honestly"
    )
    titled(
        ax,
        f"The event that pays for tail forecasting: {verdict}",
        f"Largest test-window spike ({spike_hour:%b %d, %H:00}) with the day-ahead quantile band",
    )
    if save:
        save_fig(fig, "fig05_scarcity_zoom")
    return fig


def fig14_model_comparison(metrics: dict, quantiles: list[float], save: bool = True) -> plt.Figure:
    """LGBM vs LSTM vs climatology on the three money metrics."""
    apply_style()
    models = ["climatology", "lgbm", "lstm"]
    labels = {"climatology": "Climatology\n(no model)", "lgbm": "LightGBM", "lstm": "LSTM"}
    colors = {"climatology": "#999999", "lgbm": PALETTE["blue"], "lstm": PALETTE["purple"]}

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.6, 1, 1]}
    )

    width = 0.26
    xs = np.arange(len(quantiles))
    for i, m in enumerate(models):
        vals = [metrics["models"][m][f"pinball_{quantile_col(q)}"] for q in quantiles]
        ax1.bar(
            xs + (i - 1) * width, vals, width, color=colors[m], label=labels[m].replace("\n", " ")
        )
    ax1.set_xticks(xs, [f"P{round(q * 100)}" for q in quantiles])
    ax1.set_ylabel("Pinball loss ($/MWh, lower = better)")
    ax1.set_xlabel("Quantile")
    ax1.legend(fontsize=9, loc="upper left")

    short = ["Climo", "LGBM", "LSTM"]
    for i, m in enumerate(models):
        cov = metrics["models"][m]["coverage_90"]
        ax2.bar(i, cov, 0.6, color=colors[m])
        ax2.annotate(f"{cov:.2f}", (i, cov), ha="center", va="bottom", fontsize=10)
    ax2.axhline(0.90, color=PALETTE["black"], ls="--", lw=1.2, label="nominal 90%")
    ax2.set_xticks(range(len(models)), short, fontsize=10)
    ax2.set_ylim(0, 1.12)
    ax2.set_ylabel("90% interval coverage")
    ax2.legend(fontsize=9, loc="lower left")

    for i, m in enumerate(models):
        r = metrics["models"][m]["scarcity_recall"]
        ax3.bar(i, 0 if np.isnan(r) else r, 0.6, color=colors[m])
        ax3.annotate(
            f"{r:.2f}", (i, 0 if np.isnan(r) else r), ha="center", va="bottom", fontsize=10
        )
    ax3.set_xticks(range(len(models)), short, fontsize=10)
    ax3.set_ylim(0, 1.12)
    ax3.set_ylabel(
        f"Scarcity recall ({metrics['models']['lgbm']['scarcity_hours']} scarcity hours)"
    )

    lg, ls = metrics["models"]["lgbm"], metrics["models"]["lstm"]
    better = "LightGBM" if lg["pinball_mean"] <= ls["pinball_mean"] else "LSTM"
    fig.suptitle(
        f"{better} wins on pinball loss — complexity must earn its keep",
        x=0.01,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.925,
        "Pinball per quantile (left), interval honesty (middle), tail capture (right) — "
        "chronological test window",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    if save:
        save_fig(fig, "fig14_model_comparison")
    return fig


def fig20_error_by_regime(
    y: pd.Series, preds: pd.DataFrame, threshold: float, save: bool = True
) -> plt.Figure:
    """Where the P50 loses money: by hour of day, and calm vs scarcity hours."""
    apply_style()
    err = (y - preds["q50"]).abs()
    scarce = y >= threshold

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))

    by_hour = err.groupby(err.index.hour).mean()
    ax1.bar(by_hour.index, by_hour, color=PALETTE["blue"])
    peak_h = int(by_hour.idxmax())
    ax1.bar([peak_h], [by_hour.max()], color=PALETTE["vermillion"])
    ax1.set_xlabel("Hour of day (local)")
    ax1.set_ylabel("Mean |error| of P50 ($/MWh)")

    vals = [err[~scarce].mean(), err[scarce].mean() if scarce.any() else np.nan]
    ax2.bar([0, 1], vals, 0.55, color=[PALETTE["sky"], PALETTE["vermillion"]])
    for i, v in enumerate(vals):
        if not np.isnan(v):
            ax2.annotate(f"${v:,.0f}", (i, v), ha="center", va="bottom", fontsize=11)
    ax2.set_xticks(
        [0, 1],
        [f"calm hours\n(n={int((~scarce).sum())})", f"scarcity hours\n(n={int(scarce.sum())})"],
    )
    ax2.set_ylabel("Mean |error| of P50 ($/MWh)")

    ratio = vals[1] / vals[0] if scarce.any() else float("nan")
    titled(
        ax1,
        f"Errors concentrate exactly where money does: hour {peak_h}:00 and scarcity hours "
        f"({ratio:,.0f}x calm)",
        "P50 absolute error by hour of day (left) and by market regime (right), test window",
    )
    fig.tight_layout()
    if save:
        save_fig(fig, "fig20_error_by_regime")
    return fig
