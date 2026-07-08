"""Leakage-safe feature matrix for next-day hourly price forecasting.

Why this matters: the single most common way energy-price backtests lie is
*future leakage* — features that quietly encode information unavailable when
the bid was due. Every feature here is built by mechanical lagging (`shift`)
against an explicit issue-time convention, the minimum legal lag for every
feature is declared as data (``FEATURE_SPEC``), and a test proves the rules
hold. If the honesty of this file fails, every revenue number after it is
fiction.

Issue-time convention (documented in the model card):
    The forecast for delivery day D is issued at **09:00 on D-1**, ahead of
    ERCOT's 10:00 DAM gate closure. At that moment the following are known:

    - RTM prices  : through the end of D-2  → minimum lag 48h vs any hour of D
    - DAM prices  : through delivery day D-1 (cleared ~13:30 on D-2) → min lag 24h
    - AS MCPCs    : cleared with the DAM     → min lag 24h
    - Load/wind/solar actuals: published with delay; we require ≥48h lag
      (persistence proxies — see ``data/exogenous.py`` for why forecasts
      themselves are unavailable for a historical study year)
    - Calendar    : deterministic, lag 0

Targets: ``y_price``  = next-day hourly RTM price (the dispatch driver);
         ``y_spread`` = next-day hourly DART spread (DAM - RTM).
"""

from __future__ import annotations

from dataclasses import dataclass

import holidays as holidays_lib
import numpy as np
import pandas as pd

from forecast_to_dispatch.data.exogenous import EXOG_COLUMNS


@dataclass(frozen=True)
class FeatureRule:
    """One feature's provenance: source column, transform, and minimum legal lag."""

    source: str  # column in the joined panel+exog frame, or 'calendar'
    lag_hours: int  # shift applied (0 for calendar)
    window: int = 0  # rolling window length in hours (0 = plain lag)
    agg: str = ""  # rolling aggregation ('mean'/'max'/'std')


# The authoritative feature catalog. Tests iterate this — adding a feature here
# automatically subjects it to the leakage proof.
FEATURE_SPEC: dict[str, FeatureRule] = {
    # Real-time price history (known through D-2 → lag >= 48)
    "rtm_lag48": FeatureRule("rtm_price", 48),
    "rtm_lag72": FeatureRule("rtm_price", 72),
    "rtm_lag168": FeatureRule("rtm_price", 168),
    "rtm_roll24_mean": FeatureRule("rtm_price", 48, 24, "mean"),
    "rtm_roll24_max": FeatureRule("rtm_price", 48, 24, "max"),
    "rtm_roll24_std": FeatureRule("rtm_price", 48, 24, "std"),
    "rtm_roll168_mean": FeatureRule("rtm_price", 48, 168, "mean"),
    "rtm_roll168_max": FeatureRule("rtm_price", 48, 168, "max"),
    # Day-ahead price history (D-1 cleared on D-2 → lag >= 24)
    "dam_lag24": FeatureRule("dam_price", 24),
    "dam_lag48": FeatureRule("dam_price", 48),
    "dam_lag168": FeatureRule("dam_price", 168),
    # Spread history (needs RTM → lag >= 48)
    "spread_lag48": FeatureRule("dart_spread", 48),
    "spread_roll168_mean": FeatureRule("dart_spread", 48, 168, "mean"),
    # Ancillary clearing prices (cleared with DAM → lag >= 24)
    "as_total_lag24": FeatureRule("as_total", 24),
    # Exogenous drivers (actuals used as persistence proxies → lag >= 48)
    "load_lag48": FeatureRule("load_mw", 48),
    "load_lag168": FeatureRule("load_mw", 168),
    "wind_lag48": FeatureRule("wind_mw", 48),
    "solar_lag48": FeatureRule("solar_mw", 48),
    "net_load_lag48": FeatureRule("net_load_mw", 48),
    "net_load_lag168": FeatureRule("net_load_mw", 168),
    "net_load_roll168_mean": FeatureRule("net_load_mw", 48, 168, "mean"),
    # Calendar (deterministic)
    "hour_sin": FeatureRule("calendar", 0),
    "hour_cos": FeatureRule("calendar", 0),
    "hour": FeatureRule("calendar", 0),
    "day_of_week": FeatureRule("calendar", 0),
    "month": FeatureRule("calendar", 0),
    "is_weekend": FeatureRule("calendar", 0),
    "is_holiday": FeatureRule("calendar", 0),
}

# Minimum legal lag per source, given the issue-time convention above.
MIN_LAG_BY_SOURCE = {
    "rtm_price": 48,
    "dart_spread": 48,
    "dam_price": 24,
    "as_total": 24,
    "load_mw": 48,
    "wind_mw": 48,
    "solar_mw": 48,
    "net_load_mw": 48,
    "calendar": 0,
}

TARGET_COLUMNS = ["y_price", "y_spread"]


def _calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    tx_holidays = holidays_lib.UnitedStates(subdiv="TX", years=sorted(set(index.year)))
    hour = index.hour
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "hour": hour,
            "day_of_week": index.dayofweek,
            "month": index.month,
            "is_weekend": (index.dayofweek >= 5).astype(int),
            "is_holiday": pd.Series(index.date, index=index).isin(tx_holidays).astype(int),
        },
        index=index,
    )


def build_features(
    panel: pd.DataFrame,
    exog: pd.DataFrame | None = None,
    require_targets: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (X, targets) for next-day hourly forecasting from the price panel.

    Every non-calendar feature is a lag or a rolling statistic of a *shifted*
    series, so causality is mechanical rather than asserted. Rows in the lag
    warm-up period are dropped; the function fails loudly if the result would
    be empty.
    """
    frame = panel.copy()
    frame["as_total"] = frame[["regup", "regdn", "rrs", "nspin", "ecrs"]].sum(axis=1)
    if exog is not None:
        missing = [c for c in EXOG_COLUMNS if c not in exog.columns]
        if missing:
            raise ValueError(f"Exogenous frame missing columns: {missing}")
        frame = frame.join(exog, how="left")

    X = pd.DataFrame(index=frame.index)
    for name, rule in FEATURE_SPEC.items():
        if rule.source == "calendar":
            continue
        if rule.source not in frame.columns:
            continue  # exog-less mode (e.g. tests without exogenous data)
        if rule.lag_hours < MIN_LAG_BY_SOURCE[rule.source]:
            raise ValueError(
                f"Feature {name} lags {rule.source} by {rule.lag_hours}h, below the "
                f"legal minimum {MIN_LAG_BY_SOURCE[rule.source]}h — future leakage."
            )
        shifted = frame[rule.source].shift(rule.lag_hours)
        if rule.window:
            X[name] = shifted.rolling(rule.window, min_periods=rule.window).agg(rule.agg)
        else:
            X[name] = shifted

    X = X.join(_calendar_features(frame.index))

    targets = pd.DataFrame(
        {"y_price": frame["rtm_price"], "y_spread": frame["dart_spread"]}, index=frame.index
    )

    keep = X.notna().all(axis=1)
    if require_targets:
        keep &= targets.notna().all(axis=1)
    X, targets = X.loc[keep], targets.loc[keep]
    if X.empty:
        raise ValueError("Feature matrix is empty — input panel shorter than lag warm-up (7d)")
    return X, targets


def issue_time_for(target_hour: pd.Timestamp, issue_hour: int = 9) -> pd.Timestamp:
    """The moment the forecast for ``target_hour``'s delivery day was issued:
    09:00 local on D-1. Used by the leakage test and, later, the audit log."""
    delivery_day = target_hour.normalize()
    return delivery_day - pd.Timedelta(days=1) + pd.Timedelta(hours=issue_hour)


def run(config: dict, args: object) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pipeline stage: processed panel (+ exogenous drivers) -> feature matrix.

    Live mode fetches the exogenous archives (cached); sample mode reads the
    committed exogenous sample so the whole stage stays offline.
    """
    from forecast_to_dispatch.config import resolve_path
    from forecast_to_dispatch.data.exogenous import fetch_exogenous

    processed_dir = resolve_path(config, "processed")
    panel = pd.read_parquet(processed_dir / "ercot_prices.parquet")

    if getattr(args, "sample", False):
        exog_file = resolve_path(config, "sample") / "ercot_exog_sample.parquet"
        if not exog_file.exists():
            raise FileNotFoundError(f"Committed exogenous sample missing: {exog_file}")
        exog = pd.read_parquet(exog_file)
    else:
        years = sorted({panel.index.min().year, panel.index.max().year})
        exog = pd.concat(
            [
                fetch_exogenous(y, resolve_path(config, "raw"), config["market"]["timezone"])
                for y in years
            ]
        ).sort_index()

    X, targets = build_features(panel, exog)
    X.to_parquet(processed_dir / "features.parquet")
    targets.to_parquet(processed_dir / "targets.parquet")
    print(
        f"[features] {X.shape[0]} rows x {X.shape[1]} features "
        f"({X.index.min():%Y-%m-%d} .. {X.index.max():%Y-%m-%d}); targets: {list(targets.columns)}"
    )
    return X, targets
