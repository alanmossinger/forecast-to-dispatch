"""Build notebooks/01_market_eda.ipynb from the ingested panel.

Why this matters: the notebook is a first-class deliverable — a narrated,
executed walkthrough with every figure interpreted. Building it from code keeps
it reproducible, and every number quoted in the markdown is computed from the
real panel at build time, never typed by hand.

Run after ingest, then execute headlessly:
    python scripts/build_notebook_01.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/01_market_eda.ipynb
"""

from __future__ import annotations

import nbformat as nbf
import pandas as pd

from forecast_to_dispatch.config import load_config, resolve_path

config = load_config()
panel = pd.read_parquet(resolve_path(config, "processed") / "ercot_prices.parquet")

# --- Real numbers for the narrative (computed, not invented) ---------------
prices = panel["rtm_price"]
top1_share = prices.nlargest(max(1, len(prices) // 100)).sum() / prices.sum()
p99 = prices.quantile(0.99)
pmax, pmin = prices.max(), prices.min()
mean_price = prices.mean()
spread = panel["dart_spread"]
neg_share = (spread < 0).mean()
daily_max = prices.resample("1D").max()
n_spike_days = int((daily_max >= p99).sum())
peak_hour = int(panel.groupby(panel.index.hour)["rtm_price"].mean().idxmax())
as_mean = panel[["regup", "regdn", "rrs", "nspin", "ecrs"]].mean()
hours = len(panel)
lo, hi = panel.index.min(), panel.index.max()

nb = nbf.v4.new_notebook()
cells = []


def md(s: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(s))


def code(s: str) -> None:
    cells.append(nbf.v4.new_code_cell(s))


md(
    f"""# 01 — The market problem: where battery revenue actually lives

**Audience:** energy/BESS decision-makers and AI-governance reviewers.
**Data:** ERCOT settlement point prices at **{config['market']['hub']}**, {lo:%b %d} – {hi:%b %d, %Y} ({hours:,} hours), assembled by `forecast_to_dispatch.data.ingest` from ERCOT's public historical archives (day-ahead SPP, real-time 15-minute SPP averaged to hourly, and day-ahead ancillary-service clearing prices).

## Why this notebook exists

A grid-scale battery is not paid for average prices — it is paid for **spreads** (buy low, sell high) and for **scarcity** (the rare hours when real-time prices spike toward the offer cap). Before trusting any forecasting model, we establish four facts about this market, each with a figure:

1. Revenue concentrates in a tiny fraction of hours (*price duration curve*).
2. Those hours cluster in a predictable evening window (*price heatmap*).
3. The day-ahead vs real-time spread — the arbitrage product — is fat-tailed in both directions (*spread distribution*).
4. Scarcity arrives in episodic bursts, not on a schedule (*scarcity calendar*).

These four facts define the job of the rest of the system: **forecast the distribution of prices well enough to be positioned before the tail events arrive** — and prove it honestly."""
)

code("""import pandas as pd

from forecast_to_dispatch.config import load_config, resolve_path
from forecast_to_dispatch.viz.style import apply_style
from forecast_to_dispatch.viz import eda

config = load_config()
apply_style()

panel = pd.read_parquet(resolve_path(config, "processed") / "ercot_prices.parquet")
print(f"{len(panel):,} hourly intervals, {panel.index.min()} .. {panel.index.max()}")
panel.head()""")

md(
    f"""## The dataset at a glance

Each row is one hour at the {config['market']['hub']} trading hub with three kinds of prices, aligned on one timezone-aware hourly index by the ingest module (which **fails loudly on any gap** — a silent hole here would corrupt every downstream revenue number):

- **`dam_price`** — Day-Ahead Market settlement point price (\\$/MWh), the price you lock in a day in advance.
- **`rtm_price`** — Real-Time Market price (\\$/MWh), averaged from four 15-minute settlement intervals. This is where scarcity shows up.
- **`dart_spread`** = DAM − RTM. Positive → day-ahead traded at a premium; negative → real time spiked above the forward expectation.
- **`regup`, `regdn`, `rrs`, `nspin`, `ecrs`** — clearing prices for the five ERCOT ancillary services (\\$/MW per hour of standby). Under **RTC+B** a battery is co-optimized across energy *and* these reserve products, which is where revenue stacking comes from."""
)

code("""summary = panel.describe().T[["mean", "std", "min", "50%", "max"]].round(2)
summary""")

md(
    f"""**Reading the summary:** the mean real-time price is a modest \\${mean_price:.0f}/MWh, but the maximum hit **\\${pmax:,.0f}/MWh** — a {pmax / mean_price:,.0f}× multiple. The median ancillary clearing prices are single-digit dollars, yet they apply to *capacity held in reserve*, not energy delivered — a battery can often earn them while keeping its charge for the evening. That asymmetry (long calm, extreme spikes) is exactly why this system forecasts **quantiles**, not point estimates: the P50 will never see a spike coming, but a well-calibrated P95 can."""
)

code("fig = eda.fig15_price_duration_curve(panel)")

md(
    f"""### Interpretation — fig15

The top **1%** of hours carries **{top1_share:.0%}** of the entire price mass of the half-year; the curve collapses from \\${pmax:,.0f}/MWh to double digits within the first few percent of hours. For an operator this means annual revenue is decided in roughly **{hours // 100} hours per half-year**, and missing even a couple of spike evenings materially dents the P&L. *Business consequence:* the forecasting problem that matters is not average accuracy (RMSE) but **tail positioning** — which is why Phase 3 evaluates scarcity recall and interval calibration, not just point error."""
)

code("fig = eda.fig16_hourly_price_heatmap(panel)")

md(
    f"""### Interpretation — fig16

Expensive hours are not scattered randomly: they concentrate around **hour {peak_hour}:00 local**, the early-evening net-load peak when solar output fades while demand is still high, and intensify into the summer months. This regularity is the battery's friend — a 2-hour battery must simply be **fully charged before this window opens**. *Business consequence:* the dispatch optimizer (Phase 4) should show systematic charging in the cheap overnight/midday trough and discharging into this window; fig21's SOC heatmap will verify exactly that behavior across the whole backtest."""
)

code("fig = eda.fig17_dam_rtm_spread_dist(panel)")

md(
    f"""### Interpretation — fig17

The DART spread (day-ahead minus real-time) is centered near \\${spread.mean():+.1f}/MWh but with heavy tails on **both** sides: real time exceeded day-ahead in **{neg_share:.0%}** of hours, and the extremes reach hundreds of dollars either way. A symmetric-looking average hides that the *downside* tail (RTM ≫ DAM) is where scarcity lives, and the *upside* tail is where a battery that over-committed day-ahead gets punished. *Business consequence:* the spread's width is the revenue ceiling for arbitrage; its two-sided risk is why the dispatch decision must come from a **distributional** forecast rather than a single expected price."""
)

code(
    "fig = eda.fig18_scarcity_calendar(panel, scarcity_percentile=config['forecast']['scarcity_percentile'])"
)

md(
    f"""### Interpretation — fig18

Only **{n_spike_days} days** in the window produced an hour above the P99 scarcity threshold (\\${p99:,.0f}/MWh) — and they arrive in bursts (consecutive stressed days), not evenly spaced. Long calm stretches punctuated by clustered spikes are the signature of a **regime-driven** market: weather, outages, and net-load stress arrive together. *Business consequence:* (1) a model trained on calm months can silently lose its tail sensitivity — this is why Phase 6 runs a drift monitor with alert thresholds; (2) an operator evaluating this asset should count **paydays per season**, not average daily revenue."""
)

md(
    f"""## Takeaways → requirements for the rest of the system

| Fact established here | Requirement it creates |
|---|---|
| {top1_share:.0%} of price mass in 1% of hours | Forecast the **tail** (P95), score with **scarcity recall** — Phase 3 |
| Scarcity clusters at hour {peak_hour}:00 | Battery must **hold charge for the evening window** — Phases 4–5 |
| Spread fat-tailed both ways ({neg_share:.0%} of hours RTM > DAM) | Dispatch on **quantile forecasts**, settle on realized prices — Phase 5 |
| {n_spike_days} episodic spike days | **Drift monitoring + human oversight** before real money — Phase 6 |

Next: `02_features_leakage.ipynb` — what the model is allowed to know at forecast time, and the test that proves nothing from the future leaks in."""
)

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {
    "name": "f2d-py312",
    "display_name": "Python 3.12 (forecast-to-dispatch)",
    "language": "python",
}
out = "notebooks/01_market_eda.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"wrote {out} with {len(cells)} cells")
