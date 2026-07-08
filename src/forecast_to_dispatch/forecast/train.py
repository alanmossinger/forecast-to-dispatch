"""Train, evaluate, and register the quantile forecasters.

Why this matters: a governed model is not 'a pickle somewhere' — it is a
versioned artifact with its metrics, its config snapshot, and its place in a
registry that serving reads and rollback can re-point. This module produces
exactly that: ``models/<model_id>/`` with weights + ``metrics.json`` +
``config_snapshot.yaml``, and ``models/registry.json`` naming the current
production model. The evaluation split is strictly chronological — the test
window is the *future* relative to training, never a random shuffle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from forecast_to_dispatch.forecast.base import Forecaster, quantile_col
from forecast_to_dispatch.forecast.conformal import ConformalizedForecaster
from forecast_to_dispatch.forecast.lstm import LSTMForecaster
from forecast_to_dispatch.forecast.metrics import evaluate_quantile_forecast
from forecast_to_dispatch.forecast.quantile_lgbm import QuantileLGBM


def registry_file(models_root: Path, fast: bool) -> Path:
    """The registry the current mode reads/writes: sample (fast) runs get their
    own file so CI can never re-point production to a smoke-test model."""
    return models_root / ("registry_sample.json" if fast else "registry.json")


def chronological_split(
    X: pd.DataFrame, y: pd.Series, test_fraction: float = 0.33
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split by whole delivery days, past -> train, future -> test."""
    days = X.index.normalize().unique().sort_values()
    n_test = max(1, int(len(days) * test_fraction))
    test_days = days[-n_test:]
    is_test = X.index.normalize().isin(test_days)
    return X.loc[~is_test], y.loc[~is_test], X.loc[is_test], y.loc[is_test]


def climatology_baseline(
    y_train: pd.Series, X_test: pd.DataFrame, quantiles: list[float]
) -> pd.DataFrame:
    """The 'no model' yardstick: per hour-of-day empirical quantiles of the
    training window. Any model that can't beat this has learned nothing."""
    by_hour = y_train.groupby(y_train.index.hour)
    rows = {}
    for q in quantiles:
        qs = by_hour.quantile(q)
        rows[quantile_col(q)] = X_test.index.hour.map(qs).to_numpy()
    return pd.DataFrame(rows, index=X_test.index)


def train_forecaster(
    X: pd.DataFrame,
    targets: pd.DataFrame,
    config: dict[str, Any],
    models_root: Path,
    fast: bool = False,
) -> dict[str, Any]:
    """Train LGBM + LSTM, evaluate on the chronological test window, register.

    ``fast=True`` (sample/CI mode) shrinks trees and epochs so the offline
    pipeline stays minutes-fast; the registry records which mode produced the
    artifact so a fast model can never masquerade as the studied one.
    """
    quantiles = config["forecast"]["quantiles"]
    seed = config["forecast"]["random_seed"]
    y = targets["y_price"]

    # Smoke-test artifacts live in their own (gitignored) subtree so they can
    # never be confused with — or overwrite — the studied production models.
    if fast:
        models_root = models_root / "sample"

    X_tr, y_tr, X_te, y_te = chronological_split(X, y)
    # Scarcity threshold comes from the TRAINING window only — the evaluation
    # must not peek at test-period prices to define its own yardstick.
    threshold = float(y_tr.quantile(config["forecast"]["scarcity_percentile"]))

    # Both models are conformally calibrated (see conformal.py) so their
    # stated intervals are honest out-of-sample — identical treatment keeps
    # the comparison fair.
    models: dict[str, Forecaster] = {
        "lgbm": ConformalizedForecaster(
            QuantileLGBM(quantiles, seed=seed, n_estimators=120 if fast else 500)
        ),
        "lstm": ConformalizedForecaster(
            LSTMForecaster(quantiles, seed=seed, max_epochs=25 if fast else 200)
        ),
    }

    results: dict[str, Any] = {
        "train_window": [str(X_tr.index.min()), str(X_tr.index.max())],
        "test_window": [str(X_te.index.min()), str(X_te.index.max())],
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "fast_mode": fast,
        "models": {},
    }

    baseline_preds = climatology_baseline(y_tr, X_te, quantiles)
    results["models"]["climatology"] = evaluate_quantile_forecast(
        y_te, baseline_preds, quantiles, threshold
    )

    registry_models = {}
    for name, model in models.items():
        print(f"[forecast] training {name} on {len(X_tr):,} hours ...")
        model.fit(X_tr, y_tr)
        preds = model.predict_quantiles(X_te)
        metrics = evaluate_quantile_forecast(y_te, preds, quantiles, threshold)
        results["models"][name] = metrics

        model_id = f"{name}_{pd.Timestamp(results['train_window'][1]).strftime('%Y%m%d')}"
        model_dir = models_root / model_id
        model.save(model_dir)
        (model_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(config))
        registry_models[name] = model_id
        print(
            f"[forecast] {name}: pinball={metrics['pinball_mean']:.3f} "
            f"coverage90={metrics['coverage_90']:.2f} "
            f"scarcity_recall={metrics['scarcity_recall']:.2f} -> {model_dir}"
        )

    # metrics.json is written AFTER the loop so every model dir carries the
    # COMPLETE comparison (the model card renders challenger columns from it).
    for name, model_id in registry_models.items():
        (models_root / model_id / "metrics.json").write_text(
            json.dumps({**results, "this_model": name}, indent=2)
        )

    # LGBM is the production model (registry 'current'); the LSTM is the
    # challenger kept for comparison. Rollback (Phase 6) re-points 'current'.
    # Fast/sample runs write a SEPARATE registry so a CI smoke model can never
    # silently replace the studied production pointer.
    registry_path = registry_file(models_root, fast)
    previous = None
    if registry_path.exists():
        previous = json.loads(registry_path.read_text()).get("current")
    registry = {
        "current": registry_models["lgbm"],
        "previous": previous,
        "challenger": registry_models["lstm"],
        "models": registry_models,
    }
    registry_path.write_text(json.dumps(registry, indent=2))
    results["registry"] = registry
    return results


def run(config: dict[str, Any], args: object) -> dict[str, Any]:
    """Pipeline stage: features -> trained, evaluated, registered forecasters."""
    from forecast_to_dispatch.config import resolve_path

    processed = resolve_path(config, "processed")
    X = pd.read_parquet(processed / "features.parquet")
    targets = pd.read_parquet(processed / "targets.parquet")
    results = train_forecaster(
        X,
        targets,
        config,
        models_root=resolve_path(config, "models"),
        fast=bool(getattr(args, "sample", False)),
    )
    (processed / "forecast_metrics.json").write_text(json.dumps(results, indent=2))
    return results
