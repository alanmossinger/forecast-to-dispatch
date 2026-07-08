"""Build notebooks/02_features_leakage.ipynb — the feature story + no-leakage proof.

All quoted numbers are computed from the real feature matrix at build time.
Run: python scripts/build_notebook_02.py
Then: python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=f2d-py312 notebooks/02_features_leakage.ipynb
"""

from __future__ import annotations

import nbformat as nbf
import pandas as pd

from forecast_to_dispatch.config import load_config, resolve_path
from forecast_to_dispatch.features.build import FEATURE_SPEC

config = load_config()
processed = resolve_path(config, "processed")
X = pd.read_parquet(processed / "features.parquet")
targets = pd.read_parquet(processed / "targets.parquet")
panel = pd.read_parquet(processed / "ercot_prices.parquet")

exog = pd.read_parquet(resolve_path(config, "raw") / "ercot_wind_solar_2024.parquet").join(
    pd.read_parquet(resolve_path(config, "raw") / "ercot_native_load_2024.parquet")
)
exog["net_load_mw"] = exog["load_mw"] - exog["wind_mw"] - exog["solar_mw"]

# --- computed narrative numbers --------------------------------------------
joined = panel.join(exog, how="inner")
rho_now = joined["net_load_mw"].corr(joined["rtm_price"], method="spearman")
common = X.index.intersection(targets.index)
rho_lag = X.loc[common, "net_load_lag48"].corr(targets.loc[common, "y_price"], method="spearman")
rho_dam = X.loc[common, "dam_lag24"].corr(targets.loc[common, "y_price"], method="spearman")
n_rows, n_feats = X.shape

spec_rows = "\n".join(
    f"| `{name}` | `{r.source}` | {r.lag_hours}h | "
    f"{f'{r.window}h {r.agg}' if r.window else 'lag'} |"
    for name, r in FEATURE_SPEC.items()
)

nb = nbf.v4.new_notebook()
cells = []


def md(s: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(s))


def code(s: str) -> None:
    cells.append(nbf.v4.new_code_cell(s))


md(
    """# 02 — What the model is allowed to know (and the proof it can't cheat)

**The stake:** the most common way energy-price backtests lie is *future leakage* — features that quietly encode information unavailable when the bid was actually due. Leakage inflates backtest revenue silently, and the error only surfaces when real money underperforms the study. This notebook shows the feature matrix and then **proves** its causality with a truncation experiment.

## The issue-time convention

Every forecast in this project is issued at **09:00 on D-1**, one hour before ERCOT's 10:00 Day-Ahead Market gate closure. At that moment, the market rules say the following is knowable:

| Information | Known through | Minimum legal lag vs delivery day D |
|---|---|---|
| Real-time (RTM) prices | end of D-2 | **48h** |
| Day-ahead (DAM) prices | delivery day D-1 (cleared ~13:30 on D-2) | **24h** |
| Ancillary clearing prices (MCPC) | cleared with the DAM | **24h** |
| Load / wind / solar **actuals** | published with delay → we require D-2 | **48h** |
| Calendar | deterministic | 0h |

**A deliberate honesty choice:** ERCOT's day-ahead *forecasts* of load, wind, and solar (which a production system would use) expire from the ISO's report archive and cannot be reconstructed for a 2024 study window. Rather than pretend actuals were forecastable, we use **lagged actuals as persistence proxies** — strictly weaker information than a real forecast feed. This is recorded in the model card as a limitation; live deployment would only improve on the numbers reported here."""
)

code("""import pandas as pd

from forecast_to_dispatch.config import load_config, resolve_path
from forecast_to_dispatch.features.build import build_features
from forecast_to_dispatch.viz import eda
from forecast_to_dispatch.viz.style import apply_style

config = load_config()
apply_style()
processed = resolve_path(config, "processed")

X = pd.read_parquet(processed / "features.parquet")
targets = pd.read_parquet(processed / "targets.parquet")
panel = pd.read_parquet(processed / "ercot_prices.parquet")
print(f"{X.shape[0]:,} forecastable hours x {X.shape[1]} features; targets: {list(targets.columns)}")
X.tail(3)""")

md(
    f"""## The feature catalog ({n_feats} features, every one with a declared legal lag)

The catalog is *data*, not convention: `FEATURE_SPEC` maps every feature to its source series and lag, and the test suite iterates it — a feature added below the legal minimum lag fails CI before it can contaminate a backtest.

| Feature | Source | Lag | Transform |
|---|---|---|---|
{spec_rows}

Three feature families carry the market logic: **price history** (autoregressive structure and the weekly cycle), **DAM information** (the market's own day-ahead expectation, legitimately known for D-1), and **net-load drivers** (the physics: scarcity = high demand + weak renewables)."""
)

code(
    "fig = eda.fig19_feature_target_relationships(panel, exog=_load_exog(), X=X, y=targets['y_price'])"
)

md(
    f"""### Interpretation — fig19

**Top row (market physics, same hour):** real-time price rises sharply with net load — Spearman ρ = **{rho_now:.2f}** — with the steep nonlinearity all scarcity pricing shows: flat for most of the range, then explosive in the top decile. Solar's effect is the mirror image (its evening fade is what creates the net-load peak).

**Bottom row (the model's causal view):** the same relationships seen only through leakage-safe lags. Net load 48h earlier still correlates at ρ = **{rho_lag:.2f}** with the next-day price, and the prior-day DAM price — the market's own published expectation — carries ρ = **{rho_dam:.2f}**. The signal weakens but survives.

**Business consequence:** the gap between the two rows *is* the forecasting problem — and it is why reported revenue capture will land meaningfully below the perfect-foresight ceiling. Any backtest whose capture approaches 100% should be presumed leaky."""
)

md(
    """## The no-leakage proof (truncation experiment)

Claiming causality is easy; proving it is a five-line experiment. We take one delivery day, **blind every input the market rules say was unknown at its 09:00 D-1 issue time** (all RTM prices from the issue moment onward; DAM and ancillary prices for the delivery day itself), rebuild the features from the blinded data, and compare.

If any feature saw the future, the two matrices differ and the assertion fails. This exact experiment runs in CI (`tests/test_features.py::test_truncation_at_issue_time_changes_nothing`) — the build gate will not pass with a leaky feature."""
)

code(
    """import numpy as np
from forecast_to_dispatch.features.build import issue_time_for

X_full, _ = build_features(panel)
delivery_day = X_full.index[-24].normalize()
issue = issue_time_for(delivery_day)
print(f"delivery day: {delivery_day:%Y-%m-%d}, forecast issued {issue}")

blinded = panel.copy()
blinded.loc[blinded.index >= issue, ["rtm_price", "dart_spread"]] = np.nan
blinded.loc[blinded.index >= delivery_day,
            ["dam_price", "regup", "regdn", "rrs", "nspin", "ecrs"]] = np.nan

X_blind, _ = build_features(blinded, require_targets=False)
day_idx = X_full.index[X_full.index.normalize() == delivery_day]
pd.testing.assert_frame_equal(X_full.loc[day_idx], X_blind.loc[day_idx])
print(f"PROOF: all {len(day_idx)} delivery-day rows identical with the future blinded — no leakage.")"""
)

md(
    """## Takeaways

1. Every one of the model's inputs is timestamped and provably available at bid time; the rule is enforced by CI, not by discipline.
2. The physical driver (net load) survives the honesty constraint with usable signal — the model has something real to learn.
3. Using lagged actuals instead of unavailable forecast archives makes our results a **conservative floor**: a production deployment with live ERCOT forecast feeds gets strictly better information.

Next: `03_forecasting_lgbm_vs_lstm.ipynb` — quantile forecasts, calibration honesty, scarcity recall, and what SHAP says the model actually learned."""
)

# helper cell injected near the top for exog loading inside the notebook
cells.insert(
    2,
    nbf.v4.new_code_cell("""def _load_exog():
    raw = resolve_path(config, "raw")
    exog = pd.read_parquet(raw / "ercot_wind_solar_2024.parquet").join(
        pd.read_parquet(raw / "ercot_native_load_2024.parquet")
    )
    exog["net_load_mw"] = exog["load_mw"] - exog["wind_mw"] - exog["solar_mw"]
    return exog

exog = _load_exog()
print("exogenous drivers:", list(exog.columns))"""),
)

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {
    "name": "f2d-py312",
    "display_name": "Python 3.12 (forecast-to-dispatch)",
    "language": "python",
}
out = "notebooks/02_features_leakage.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"wrote {out} with {len(cells)} cells")
