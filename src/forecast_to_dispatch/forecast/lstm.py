"""LSTMForecaster: a sequence model behind the same quantile interface.

Why this matters: the LGBM baseline treats every hour independently; an LSTM
reads a delivery day as a 24-step sequence and can, in principle, learn
intra-day structure (the morning ramp foreshadowing the evening peak) that
tabular trees can't see. It exists to answer a governance-relevant question
with evidence instead of fashion: *does the deep model earn its complexity?*
fig14 settles that with pinball loss, coverage, and scarcity recall side by
side. Same `fit` / `predict_quantiles` contract, so dispatch and backtest code
cannot tell the two models apart.

Training details: quantile (pinball) loss summed over the five quantiles;
features standardized with train-window statistics; early stopping on the last
20% of training days; fully seeded for reproducibility.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from forecast_to_dispatch.forecast.base import Forecaster, enforce_monotone

HOURS_PER_DAY = 24


class _QuantileLSTM(nn.Module):
    def __init__(self, n_features: int, n_quantiles: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 24, F) -> (B, 24, Q)
        out, _ = self.lstm(x)
        return self.head(out)


def _pinball(pred: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    diff = target.unsqueeze(-1) - pred  # (B, 24, Q)
    return torch.mean(torch.maximum(quantiles * diff, (quantiles - 1) * diff))


class LSTMForecaster(Forecaster):
    def __init__(
        self,
        quantiles: list[float],
        seed: int = 42,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        batch_days: int = 16,
        max_epochs: int = 200,
        patience: int = 15,
    ):
        super().__init__(quantiles, seed)
        self.hidden, self.layers, self.dropout = hidden, layers, dropout
        self.lr, self.batch_days = lr, batch_days
        self.max_epochs, self.patience = max_epochs, patience
        self.model: _QuantileLSTM | None = None
        self.feature_names: list[str] = []
        self.mu: np.ndarray | None = None
        self.sigma: np.ndarray | None = None

    # -- day blocking ---------------------------------------------------------

    @staticmethod
    def _to_days(X: pd.DataFrame, y: pd.Series | None = None):
        """Group the hourly matrix into contiguous 24h delivery-day blocks.
        Incomplete days (lag warm-up edges) are dropped with a notice — the
        LSTM's unit of prediction is the full delivery day."""
        days = X.index.normalize()
        complete = days.value_counts() == HOURS_PER_DAY
        keep_days = complete[complete].index
        mask = pd.Index(days).isin(keep_days)
        Xk = X.loc[mask]
        order = np.argsort(Xk.index)
        Xk = Xk.iloc[order]
        n_days = len(Xk) // HOURS_PER_DAY
        xb = Xk.to_numpy(dtype=np.float32).reshape(n_days, HOURS_PER_DAY, -1)
        idx = Xk.index
        if y is None:
            return xb, idx
        yk = y.loc[Xk.index].to_numpy(dtype=np.float32).reshape(n_days, HOURS_PER_DAY)
        return xb, yk, idx

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LSTMForecaster":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.feature_names = list(X.columns)

        xb, yb, _ = self._to_days(X, y)
        self.mu = xb.reshape(-1, xb.shape[-1]).mean(axis=0)
        self.sigma = xb.reshape(-1, xb.shape[-1]).std(axis=0) + 1e-8
        xb = (xb - self.mu) / self.sigma
        # Price targets are heavy-tailed; train in log1p space (documented in
        # the model card) so a single $3,000 hour doesn't dominate the loss.
        yb = np.log1p(np.clip(yb, -0.99e2, None) + 100.0)  # shift keeps negatives finite

        n_val = max(1, int(0.2 * len(xb)))
        xt, yt = torch.tensor(xb[:-n_val]), torch.tensor(yb[:-n_val])
        xv, yv = torch.tensor(xb[-n_val:]), torch.tensor(yb[-n_val:])

        self.model = _QuantileLSTM(
            xb.shape[-1], len(self.quantiles), self.hidden, self.layers, self.dropout
        )
        qs = torch.tensor(self.quantiles, dtype=torch.float32)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_val, best_state, bad = float("inf"), None, 0
        for _ in range(self.max_epochs):
            self.model.train()
            perm = torch.randperm(len(xt))
            for i in range(0, len(xt), self.batch_days):
                sel = perm[i : i + self.batch_days]
                opt.zero_grad()
                loss = _pinball(self.model(xt[sel]), yt[sel], qs)
                loss.backward()
                opt.step()
            self.model.eval()
            with torch.no_grad():
                val = _pinball(self.model(xv), yv, qs).item()
            if val < best_val - 1e-5:
                best_val, bad = val, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("LSTMForecaster.predict_quantiles called before fit/load")
        if list(X.columns) != self.feature_names:
            raise ValueError("Feature columns differ from training — refusing to predict")
        xb, idx = self._to_days(X)
        if len(idx) < len(X):
            raise ValueError(
                "LSTM requires complete 24h delivery days; input contains partial days"
            )
        xb = (xb - self.mu) / self.sigma
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.tensor(xb)).numpy()  # (D, 24, Q) in log space
        out = np.expm1(out) - 100.0  # invert the log1p(+100) transform
        preds = pd.DataFrame(
            out.reshape(-1, len(self.quantiles)), index=idx, columns=self.quantile_columns
        )
        return enforce_monotone(preds)

    # -- persistence ----------------------------------------------------------

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out_dir / "lstm_state.pt")
        (out_dir / "meta.json").write_text(
            json.dumps(
                {
                    "type": "LSTMForecaster",
                    "quantiles": self.quantiles,
                    "seed": self.seed,
                    "hidden": self.hidden,
                    "layers": self.layers,
                    "dropout": self.dropout,
                    "feature_names": self.feature_names,
                    "mu": self.mu.tolist(),
                    "sigma": self.sigma.tolist(),
                }
            )
        )

    @classmethod
    def load(cls, model_dir: Path) -> "LSTMForecaster":
        meta = json.loads((model_dir / "meta.json").read_text())
        inst = cls(
            meta["quantiles"],
            seed=meta["seed"],
            hidden=meta["hidden"],
            layers=meta["layers"],
            dropout=meta["dropout"],
        )
        inst.feature_names = meta["feature_names"]
        inst.mu = np.array(meta["mu"], dtype=np.float32)
        inst.sigma = np.array(meta["sigma"], dtype=np.float32)
        inst.model = _QuantileLSTM(
            len(inst.feature_names), len(inst.quantiles), inst.hidden, inst.layers, inst.dropout
        )
        inst.model.load_state_dict(torch.load(model_dir / "lstm_state.pt", weights_only=True))
        inst.model.eval()
        return inst
