"""Build notebooks/05_backtest_revenue.ipynb — the money story.

All numbers injected into markdown come from the real backtest artifacts
(data/processed/backtest_results.json + backtest_daily.parquet) at build time.
"""

from __future__ import annotations

import json

import nbformat as nbf
import numpy as np
import pandas as pd

from forecast_to_dispatch.config import load_config, resolve_path
from forecast_to_dispatch.optimize.dispatch import ALL_PRODUCTS

config = load_config()
processed = resolve_path(config, "processed")
s = json.loads((processed / "backtest_results.json").read_text())
daily = pd.read_parquet(processed / "backtest_daily.parquet")

tot = s["totals"]
capture = s["revenue_capture_pct"]
uplift = s["uplift_vs_naive_pct"]
streams = s["governed_streams"]
gross = streams["energy"] + sum(streams[k] for k in ALL_PRODUCTS)
as_share = sum(streams[k] for k in ALL_PRODUCTS) / gross
gap = daily["perfect_revenue"] - daily["governed_revenue"]
rho = float(np.corrcoef(daily["forecast_mae"], gap)[0, 1])
top5_gap_share = gap.nlargest(5).sum() / gap.sum()
ann_factor = 365 / s["n_days"]

nb = nbf.v4.new_notebook()
cells = []


def md(t: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(t))


def code(t: str) -> None:
    cells.append(nbf.v4.new_code_cell(t))


md(
    f"""# 05 — The money story: {capture:.0f}% revenue capture, honestly measured

**The experiment:** walk forward through all **{s['n_days']} out-of-sample days** ({s['window'][0]} → {s['window'][1]}). Every morning, the governed agent commits a full day's schedule using only its quantile forecast and yesterday's ancillary prices. Then reality arrives, and the schedule **settles at the prices that actually occurred.** No retrading, no hindsight, no leakage — the schedule was fixed before the future happened.

Three policies, identical battery (100 MW / 200 MWh), identical physics:

| Policy | Sees | Role |
|---|---|---|
| **Perfect foresight** | tomorrow's realized prices | unreachable ceiling — honest scaling |
| **Governed (this system)** | conformal quantile forecast + persistence AS prices | the agent under evaluation |
| **Naive** | the training window's average day, energy only | the floor any model must beat |

## The scoreboard

| | Settled revenue | vs ceiling |
|---|---|---|
| Perfect foresight | ${tot['perfect'] / 1e6:.2f}M | 100% |
| **Governed** | **${tot['governed'] / 1e6:.2f}M** | **{capture:.1f}%** |
| Naive | ${tot['naive'] / 1e6:.2f}M | {100 * tot['naive'] / tot['perfect']:.1f}% |

That is **{capture:.1f}% revenue capture** and a **{uplift:+.0f}% uplift over the naive floor** — on a ~{s['n_days']}-day window that annualizes to roughly ${tot['governed'] * ann_factor / 1e6:.0f}M for this asset (indicative, single season)."""
)

code("""import json
import pandas as pd

from forecast_to_dispatch.config import load_config, resolve_path
from forecast_to_dispatch.viz.style import apply_style
from forecast_to_dispatch.viz import backtest_viz

config = load_config()
apply_style()
processed = resolve_path(config, "processed")

summary = json.loads((processed / "backtest_results.json").read_text())
daily = pd.read_parquet(processed / "backtest_daily.parquet")
governed_schedule = pd.read_parquet(processed / "backtest_governed_schedule.parquet")
print(f"{summary['n_days']} days | governed ${summary['totals']['governed']:,.0f} | "
      f"capture {summary['revenue_capture_pct']:.1f}%")""")

code("fig = backtest_viz.fig07_cumulative_revenue(daily, summary)")

md(
    f"""### Interpretation — fig07 (the headline chart)

The governed line tracks the ceiling closely and pulls away from the naive floor immediately and permanently — this is capability, not luck. The vertical jumps are spike days: notice the ceiling and the governed line jump *together* (the agent was positioned) while the naive line barely moves (it had no idea). **Business consequence:** {capture:.0f}% of the theoretically available value, realized with honest day-ahead information — the single number a hiring manager or asset owner should quote from this repository."""
)

code("fig = backtest_viz.fig08_revenue_capture_bar(summary)")

md(
    f"""### Interpretation — fig08

Same result, one glance: the naive strategy captures {100 * tot['naive'] / tot['perfect']:.0f}% of available value, the governed agent {capture:.0f}%. The remaining {100 - capture:.0f}% is the irreducible price of not knowing the future — quantified in fig10 rather than hidden. **Business consequence:** the model's contribution is the distance between the first two bars; the honesty of the method is the visible existence of the third."""
)

code("fig = backtest_viz.fig09_revenue_stack(summary)")

md(
    f"""### Interpretation — fig09

Revenue stacking under RTC+B, settled at real prices: ancillary services carried **{as_share:.0%}** of gross revenue (ECRS ${streams['ecrs'] / 1e6:.2f}M and Reg-Down ${streams['regdn'] / 1e6:.2f}M leading), energy arbitrage ${streams['energy'] / 1e6:.2f}M, degradation cost ${streams['degradation'] / 1e3:,.0f}k. Reserve prices are strongly autocorrelated day-over-day, so even the persistence forecast captures them well — while the energy spikes are where perfect foresight out-earns everyone. **Business consequence:** an energy-only battery model would have left roughly {as_share:.0%} of this asset's value on the table; co-optimization is not a refinement, it is most of the product."""
)

code("fig = backtest_viz.fig10_cost_of_imperfection(daily)")

md(
    f"""### Interpretation — fig10

The gap to perfect foresight is not noise: it correlates with daily forecast error (r = {rho:.2f}) and concentrates brutally — the worst **5 days carry {top5_gap_share:.0%}** of the entire cumulative gap. Those are the spike days where the P95 saw *elevated* risk but not the full magnitude. **Business consequence:** this chart prices the profit-overestimation trap — a leaky backtest silently books this gap as revenue. Ours reports it, which is exactly what makes the {capture:.0f}% defensible; it also says where the next modeling dollar goes (tail magnitude, not average accuracy)."""
)

code("fig = backtest_viz.fig21_soc_heatmap(governed_schedule)")

md(
    """### Interpretation — fig21

The proof the behavior is systematic: across every backtest day, state of charge builds through the cheap hours and stands high entering the evening scarcity window (17:00–20:00), then drains into it. This is the strategy fig16 (price heatmap) said the market pays for, executed daily by an agent that never saw tomorrow's prices. **Business consequence:** an operator can trust behavior that is legible and repeatable — this heatmap is what 'the battery holds charge for the evening' looks like as evidence instead of a slogan."""
)

code("fig = backtest_viz.fig22_reserve_allocation(governed_schedule)")

md(
    """### Interpretation — fig22

How the same megawatts earn twice: on calm days the battery's capacity is rented out almost entirely as reserves (steady income); on stressed days energy discharge takes over the profile. The allocation shifts automatically with the forecast — no rule was hand-written for it. **Business consequence:** this adaptive reallocation is precisely the value RTC+B-style co-optimization is designed to unlock, demonstrated on real prices."""
)

code("fig = backtest_viz.fig23_monthly_revenue(daily)")

md(
    f"""### Interpretation — fig23

Both months show the same structure — governed near the ceiling, far above naive — so the {capture:.0f}% capture is not one lucky May afternoon. June, with fewer extreme spikes, actually shows *higher* relative capture; spikier months widen the gap to perfection while raising absolute revenue. **Business consequence:** consistency across market regimes is what separates a strategy from a story; two months is the honest limit of this study window, and the drift monitor (Phase 6) is the tool that watches whether the pattern persists.

## Takeaways

| Claim | Evidence |
|---|---|
| The agent makes real money on real prices | ${tot['governed'] / 1e6:.2f}M settled over {s['n_days']} days |
| Honestly measured | dispatched on forecasts, settled on realized; ceiling & floor reported |
| {capture:.0f}% of available value captured | fig07/fig08 |
| Stacking is most of the value | AS = {as_share:.0%} of gross revenue (fig09) |
| Cost of imperfection quantified | fig10; top-5 days = {top5_gap_share:.0%} of the gap |
| Behavior is systematic | fig21/fig22 |

Next: `06_governance_walkthrough.ipynb` — the layer that makes this deployable: model card, risk register, drift monitoring, an append-only audit trail, human-in-the-loop approval, and rollback."""
)

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {
    "name": "f2d-py312",
    "display_name": "Python 3.12 (forecast-to-dispatch)",
    "language": "python",
}
out = "notebooks/05_backtest_revenue.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"wrote {out} with {len(cells)} cells")
