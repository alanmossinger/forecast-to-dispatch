# Model card — ERCOT price-quantile forecaster (production)

*Generated 2026-07-08 04:40 UTC from `models/registry.json`
and the registered run artifacts. Do not edit by hand — regenerate via
`python -m forecast_to_dispatch.governance.check --refresh-artifacts`.*

**Model version:** `lgbm_20240503` (challenger: `lstm_20240503`,
previous: `none`)

## Intended use

Day-ahead probabilistic forecasting of hourly ERCOT real-time settlement point
prices at HB_HOUSTON, issued 09:00 D-1, to drive co-optimized
energy + ancillary-service dispatch of a 100 MW /
200 MWh battery. **Not** for financial trading advice, other
hubs/ISOs without revalidation, or intraday re-forecasting.

## Training data

Public ERCOT archives via `gridstatus` (no paid/proprietary data): DAM and RTM
settlement point prices, DAM ancillary clearing prices (Reg Up/Down, RRS,
Non-Spin, ECRS), hourly system load, and wind/solar generation.
Training window: 2024-01-09 → 2024-05-03
(2,760 hourly observations, 28 leakage-safe features with
declared minimum legal lags).

## Evaluation protocol

Chronological split (test = strictly future, 1,368 hours);
pinball loss (proper score), central-interval coverage, and scarcity recall
(P95 vs the training-window P99 threshold of
$166/MWh). Intervals are width-scaled conformally
calibrated on the last quarter of the training window.

## Performance

| Metric | This model (LGBM+CQR) | Climatology | LSTM challenger |
|---|---|---|---|
| Pinball (mean) | **7.54** | 8.67 | 10.26 |
| 90% interval coverage | **0.86** | 0.85 | 0.76 |
| P50 MAE ($/MWh) | **18.6** | 21.9 | — |

Walk-forward backtest (57 out-of-sample days, 2024-05-04 → 2024-06-29): settled revenue $3,313,731 = **86.1% revenue capture** vs the perfect-foresight ceiling; +447% vs the naive floor. Ancillary services carried 96% of gross revenue.

## Tail behavior

Scarcity recall 0.35 on 17 realized
scarcity hours: the P95 flagged 35% of hours that
settled above the scarcity threshold. The model reliably calls the *timing*
of stressed evenings but under-calls extreme *magnitudes* (see fig05, fig10);
dispatch positioning is therefore driven by the full quantile fan, not P50.

## Limitations

- **Exogenous drivers are persistence proxies** (lagged actual load/wind/solar):
  ERCOT's historical day-ahead forecast archives expire and cannot be
  reconstructed. Production would consume live forecast feeds — reported
  results are a conservative floor.
- **Single-season study** (Jan–Jun 2024, one hub). Summer regimes and other
  hubs require revalidation; the drift monitor is the tripwire.
- **AS prices forecast by persistence** (prior-day clearing prices), and
  reserves settle on capacity without deployment modeling.
- **Price-taker assumption**: valid for 100 MW at a liquid hub, not for a
  fleet.
- LSTM challenger underperforms at this data volume; retained in the registry
  for future comparison, not serving.

## Governance controls

Leakage truncation proof, conformal coverage gate (|coverage − 0.90| ≤
0.08), PSI/KS drift monitor
(thresholds 0.2 /
0.01), append-only
hash-chained audit log, human-in-the-loop approval bound to schedule hashes,
versioned registry with one-command rollback. All enforced by
`python -m forecast_to_dispatch.governance.check` (CI release gate).

## Ownership

Model owner: forecasting lead (this repository's maintainer). Dispatch owner:
optimization lead. Operations owner: on-call operator with HITL approval
authority. Escalation: any gate failure blocks release; rollback playbook in
`governance/rollback.py`.
