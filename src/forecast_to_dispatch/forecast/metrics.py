"""Forecast quality metrics that map to money: pinball, coverage, scarcity recall.

Why this matters: RMSE rewards models that predict the boring average and
ignore the spikes that pay for the battery. These three metrics each guard a
failure mode with a dollar cost:

- **Pinball loss** (per quantile): the proper scoring rule for quantiles —
  a model can't game it by being overconfident or vague.
- **Interval coverage**: honesty about uncertainty. If the '90% interval'
  contains reality only 70% of the time, every risk decision downstream is
  built on a lie, and the battery will be under-positioned for tails.
- **Scarcity recall**: of the hours that turned out to be scarcity (top
  percentile of realized prices), in what fraction did the P95 forecast raise
  its hand above the scarcity threshold? This is tail positioning — the single
  number that says whether the model can see paydays coming.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecast_to_dispatch.forecast.base import quantile_col


def pinball_loss(y: pd.Series, pred: pd.Series, q: float) -> float:
    diff = y.to_numpy() - pred.to_numpy()
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def interval_coverage(y: pd.Series, lo: pd.Series, hi: pd.Series) -> float:
    """Share of hours where reality landed inside [lo, hi]."""
    return float(((y >= lo) & (y <= hi)).mean())


def scarcity_recall(y: pd.Series, p95: pd.Series, threshold: float) -> tuple[float, int]:
    """(recall, n_scarcity_hours): among realized-scarcity hours, share where
    the P95 forecast exceeded the threshold too. Returns NaN recall if the
    evaluation window contained no scarcity hours (reported, never hidden)."""
    scarce = y >= threshold
    n = int(scarce.sum())
    if n == 0:
        return float("nan"), 0
    return float((p95[scarce] >= threshold).mean()), n


def evaluate_quantile_forecast(
    y: pd.Series,
    preds: pd.DataFrame,
    quantiles: list[float],
    scarcity_threshold: float,
) -> dict:
    """Full metric card for one model on one evaluation window."""
    metrics: dict = {"n_hours": int(len(y))}
    for q in quantiles:
        metrics[f"pinball_{quantile_col(q)}"] = pinball_loss(y, preds[quantile_col(q)], q)
    metrics["pinball_mean"] = float(
        np.mean([metrics[f"pinball_{quantile_col(q)}"] for q in quantiles])
    )
    if 0.05 in quantiles and 0.95 in quantiles:
        metrics["coverage_90"] = interval_coverage(y, preds["q05"], preds["q95"])
    if 0.25 in quantiles and 0.75 in quantiles:
        metrics["coverage_50"] = interval_coverage(y, preds["q25"], preds["q75"])
    metrics["mae_p50"] = float((y - preds["q50"]).abs().mean())
    recall, n_scarce = scarcity_recall(y, preds["q95"], scarcity_threshold)
    metrics["scarcity_recall"] = recall
    metrics["scarcity_hours"] = n_scarce
    metrics["scarcity_threshold"] = float(scarcity_threshold)
    return metrics
