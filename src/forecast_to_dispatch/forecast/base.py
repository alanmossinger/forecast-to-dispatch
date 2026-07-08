"""The Forecaster interface: every model speaks quantiles, or it doesn't ship.

Why this matters: the dispatch optimizer consumes a *distribution* of prices,
not a point estimate — a single expected price can never position the battery
for the scarcity tail that carries most of the revenue. Forcing every model
(LightGBM today, LSTM tomorrow, anything later) behind one `fit` /
`predict_quantiles` contract makes models swappable without touching dispatch,
backtest, or governance code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


def quantile_col(q: float) -> str:
    """Canonical column name for a quantile, e.g. 0.05 -> 'q05'."""
    return f"q{round(q * 100):02d}"


def enforce_monotone(preds: pd.DataFrame) -> pd.DataFrame:
    """Sort each row's quantile predictions so q05 <= q25 <= ... <= q95.

    Why this matters: independently trained quantile models can 'cross'
    (P25 above P50), which is statistically incoherent and would confuse the
    dispatch optimizer's risk logic. Row-wise sorting is the standard,
    loss-neutral repair.
    """
    values = np.sort(preds.to_numpy(), axis=1)
    return pd.DataFrame(values, index=preds.index, columns=preds.columns)


class Forecaster(ABC):
    """Contract: fit on a leakage-safe feature matrix, emit quantile forecasts."""

    def __init__(self, quantiles: list[float], seed: int = 42):
        if sorted(quantiles) != list(quantiles):
            raise ValueError(f"Quantiles must be sorted ascending, got {quantiles}")
        self.quantiles = list(quantiles)
        self.seed = seed

    @property
    def quantile_columns(self) -> list[str]:
        return [quantile_col(q) for q in self.quantiles]

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Forecaster":
        """Train on features X and target y (next-day hourly RTM price)."""

    @abstractmethod
    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame indexed like X with one column per quantile,
        guaranteed monotone across quantiles."""
