"""QuantileLGBM: the production baseline — one gradient-boosted model per quantile.

Why this matters: LightGBM with the quantile (pinball) objective is the
workhorse of operational price forecasting — robust to the wild outliers of
power prices (tree splits don't care about magnitude), fast enough to retrain
daily, and directly explainable with SHAP. Training five separate models keeps
each quantile's objective clean; row-wise sorting repairs any crossing.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd

from forecast_to_dispatch.forecast.base import Forecaster, enforce_monotone, quantile_col

DEFAULT_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 30,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "verbosity": -1,
}


class QuantileLGBM(Forecaster):
    """One LGBMRegressor per quantile, all seeded for reproducibility."""

    def __init__(self, quantiles: list[float], seed: int = 42, **overrides):
        super().__init__(quantiles, seed)
        self.params = {**DEFAULT_PARAMS, **overrides}
        self.models: dict[float, lgb.LGBMRegressor] = {}
        self.feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "QuantileLGBM":
        self.feature_names = list(X.columns)
        for q in self.quantiles:
            model = lgb.LGBMRegressor(
                objective="quantile", alpha=q, random_state=self.seed, **self.params
            )
            model.fit(X, y)
            self.models[q] = model
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("QuantileLGBM.predict_quantiles called before fit/load")
        if list(X.columns) != self.feature_names:
            raise ValueError("Feature columns differ from training — refusing to predict")
        preds = pd.DataFrame(
            {quantile_col(q): m.predict(X) for q, m in self.models.items()}, index=X.index
        )
        return enforce_monotone(preds[self.quantile_columns])

    # -- persistence (the model registry stores these artifacts) -------------

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for q, model in self.models.items():
            joblib.dump(model, out_dir / f"lgbm_{quantile_col(q)}.joblib")
        (out_dir / "meta.json").write_text(
            json.dumps(
                {
                    "type": "QuantileLGBM",
                    "quantiles": self.quantiles,
                    "seed": self.seed,
                    "params": self.params,
                    "feature_names": self.feature_names,
                }
            )
        )

    @classmethod
    def load(cls, model_dir: Path) -> "QuantileLGBM":
        meta = json.loads((model_dir / "meta.json").read_text())
        inst = cls(meta["quantiles"], seed=meta["seed"], **meta["params"])
        inst.feature_names = meta["feature_names"]
        for q in inst.quantiles:
            inst.models[q] = joblib.load(model_dir / f"lgbm_{quantile_col(q)}.joblib")
        return inst
