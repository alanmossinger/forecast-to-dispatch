"""Drift monitor: detect market-regime shift before it costs money.

Why this matters: fig18 showed ERCOT scarcity is regime-driven — a model
trained in a calm spring can silently lose its tail sensitivity when summer
stress arrives. Two complementary detectors run on every monitored series:

- **PSI** (Population Stability Index): magnitude of distribution shift in
  interpretable bins. Industry convention: <0.1 stable, 0.1-0.2 watch,
  >0.2 act — our gate threshold (config: governance.drift.psi_threshold).
- **KS test**: distribution-free evidence that the two samples differ
  (gate on p-value < governance.drift.ks_pvalue_threshold).

Monitored: the model's key input features AND its own prediction distribution
(a model whose outputs drift is drifting, whatever its inputs say). The gate
verdict feeds ``governance.check`` — sustained drift blocks release.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# The features whose drift most directly threatens revenue (price regime and
# the net-load physics), plus the model's own output distribution.
MONITORED_FEATURES = ["rtm_lag48", "rtm_roll24_std", "dam_lag24", "net_load_lag48"]


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index with reference-quantile bins."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # degenerate/constant reference
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_frac = np.histogram(reference, edges)[0] / len(reference)
    cur_frac = np.histogram(current, edges)[0] / len(current)
    ref_frac = np.clip(ref_frac, 1e-6, None)
    cur_frac = np.clip(cur_frac, 1e-6, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str],
    psi_threshold: float,
    ks_pvalue_threshold: float,
) -> pd.DataFrame:
    """Per-series drift verdicts. A series fails if PSI exceeds the threshold
    AND the KS test corroborates (p below threshold) — requiring both avoids
    alarming on one noisy statistic."""
    rows = []
    for col in columns:
        ref, cur = reference[col].dropna().to_numpy(), current[col].dropna().to_numpy()
        p = psi(ref, cur)
        ks_p = float(stats.ks_2samp(ref, cur).pvalue)
        rows.append(
            {
                "series": col,
                "psi": round(p, 4),
                "ks_pvalue": ks_p,
                "psi_alert": p > psi_threshold,
                "ks_alert": ks_p < ks_pvalue_threshold,
                "drift": bool(p > psi_threshold and ks_p < ks_pvalue_threshold),
            }
        )
    return pd.DataFrame(rows).set_index("series")


def rolling_psi(
    reference: pd.DataFrame,
    stream: pd.DataFrame,
    columns: list[str],
    freq: str = "7D",
) -> pd.DataFrame:
    """PSI of each rolling window vs the fixed reference — fig11's time axis."""
    out: dict[str, dict] = {}
    for window_start, chunk in stream.groupby(pd.Grouper(freq=freq)):
        if len(chunk) < 48:  # too little data for a stable histogram
            continue
        out[window_start] = {
            col: psi(reference[col].dropna().to_numpy(), chunk[col].dropna().to_numpy())
            for col in columns
        }
    return pd.DataFrame(out).T
