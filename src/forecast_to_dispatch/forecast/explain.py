"""SHAP explainability of the forecaster, translated into market language.

Why this matters: an autonomous agent allocating real capital cannot be a
black box — an operator, a risk committee, and (under the EU AI Act) an
auditor must be able to ask *why* the model expects a high price tomorrow
evening. SHAP on the P50 LightGBM model attributes every prediction to its
drivers; the labels below translate raw feature names into the vocabulary the
audience actually speaks (net load, solar fade, price momentum). Note for
readers: tail-quantile models (P95) weight scarcity drivers even more heavily
than the median model shown here.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import shap

from forecast_to_dispatch.forecast.quantile_lgbm import QuantileLGBM
from forecast_to_dispatch.viz.style import apply_style, save_fig

# Raw feature name -> what it means in the market.
MARKET_LABELS = {
    "rtm_lag48": "RT price, 2 days prior (same hour)",
    "rtm_lag72": "RT price, 3 days prior (same hour)",
    "rtm_lag168": "RT price, 1 week prior (same hour)",
    "rtm_roll24_mean": "RT price, last known day avg",
    "rtm_roll24_max": "RT price, last known day max",
    "rtm_roll24_std": "RT price volatility, last known day",
    "rtm_roll168_mean": "RT price, trailing week avg",
    "rtm_roll168_max": "RT price, trailing week max",
    "dam_lag24": "DA price for prior day (market's own expectation)",
    "dam_lag48": "DA price, 2 days prior",
    "dam_lag168": "DA price, 1 week prior",
    "spread_lag48": "DA-RT spread, 2 days prior",
    "spread_roll168_mean": "DA-RT spread, trailing week avg",
    "as_total_lag24": "Ancillary clearing prices, prior day (reserve scarcity signal)",
    "load_lag48": "System load, 2 days prior",
    "load_lag168": "System load, 1 week prior",
    "wind_lag48": "Wind output, 2 days prior",
    "solar_lag48": "Solar output, 2 days prior",
    "net_load_lag48": "Net load (demand minus renewables), 2 days prior",
    "net_load_lag168": "Net load, 1 week prior",
    "net_load_roll168_mean": "Net load, trailing week avg",
    "hour_sin": "Time of day (cyclical)",
    "hour_cos": "Time of day (cyclical, phase)",
    "hour": "Hour of day",
    "day_of_week": "Day of week",
    "month": "Month",
    "is_weekend": "Weekend",
    "is_holiday": "Holiday (TX)",
}


def shap_values_p50(model: QuantileLGBM, X: pd.DataFrame) -> shap.Explanation:
    """SHAP values of the median (P50) model — the market's 'expected price'."""
    booster = model.models[0.5]
    explainer = shap.TreeExplainer(booster)
    explanation = explainer(X)
    explanation.feature_names = [MARKET_LABELS.get(c, c) for c in X.columns]
    return explanation


def fig03_shap_beeswarm(explanation: shap.Explanation, save: bool = True) -> plt.Figure:
    """Beeswarm: direction + magnitude of every driver, every hour."""
    apply_style()
    fig = plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(explanation, max_display=14, show=False)
    ax = plt.gca()
    ax.set_title(
        "Price momentum and net-load stress drive the forecast",
        loc="left",
        fontsize=14,
        fontweight="bold",
        pad=18,
    )
    ax.text(
        0,
        1.01,
        "SHAP values of the P50 LightGBM model, test window (market-language labels)",
        transform=ax.transAxes,
        fontsize=10,
        color="#555555",
        va="bottom",
    )
    fig = plt.gcf()
    fig.tight_layout()
    if save:
        save_fig(fig, "fig03_shap_beeswarm")
    return fig


def fig04_shap_bar(explanation: shap.Explanation, save: bool = True) -> plt.Figure:
    """Global importance ranking (mean |SHAP|)."""
    apply_style()
    fig = plt.figure(figsize=(9, 6.5))
    shap.plots.bar(explanation, max_display=14, show=False)
    ax = plt.gca()
    ax.set_title(
        "What the model actually relies on, ranked",
        loc="left",
        fontsize=14,
        fontweight="bold",
        pad=18,
    )
    ax.text(
        0,
        1.01,
        "Mean |SHAP| of the P50 model — average $/MWh moved per feature",
        transform=ax.transAxes,
        fontsize=10,
        color="#555555",
        va="bottom",
    )
    fig = plt.gcf()
    fig.tight_layout()
    if save:
        save_fig(fig, "fig04_shap_bar")
    return fig
