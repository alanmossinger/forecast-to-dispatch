# Results — the market story in figures

*One narrative: the market problem → how well we forecast (especially the tails) → the revenue the governed agent captures → proof it is governed. Every figure below is embedded with a 3–5 sentence interpretation ending in a business consequence. Figures are produced by real runs on real data — no number here is invented.*

> Status: populated phase by phase. Sections appear as their figures land.

---

## 1. The market problem

*Data: ERCOT settlement point prices at HB_HOUSTON, Jan 1 – Jun 29 2024 (4,343 hours), from ERCOT's public historical archives via `gridstatus` — day-ahead (DAM) hourly SPP, real-time (RTM) 15-minute SPP averaged to hourly, and DAM ancillary-service clearing prices (Reg Up/Down, RRS, Non-Spin, ECRS).*

![Price duration curve](figures/fig15_price_duration_curve.png)

The real-time price averaged just $27/MWh over the half-year, but the top **1% of hours carried 18% of the entire price mass**, peaking at $3,049/MWh — a 112× multiple of the mean. The curve collapses to double digits within the first few percent of hours, which means a battery's annual P&L is decided in roughly two days' worth of scattered hours. **Business consequence:** the forecasting problem that pays is not average accuracy but tail positioning — this is why the system is evaluated on scarcity recall and interval calibration, not RMSE.

![Hourly price heatmap](figures/fig16_hourly_price_heatmap.png)

Expensive hours concentrate around **hour 20:00 local** — the early-evening net-load peak when solar fades while demand persists — and intensify toward summer. This regularity is the battery's friend: a 2-hour battery simply must be fully charged before this window opens. **Business consequence:** the dispatch optimizer should charge in the overnight/midday trough and discharge into this window; the backtest SOC heatmap (fig21) will verify that behavior systematically.

![DART spread distribution](figures/fig17_dam_rtm_spread_dist.png)

The day-ahead-minus-real-time (DART) spread averaged only **+$3.3/MWh** but with a $63/MWh standard deviation and heavy tails both ways; real time exceeded day-ahead in **43% of hours**. The near-zero mean hides that the downside tail (RTM ≫ DAM) is where scarcity lives and the upside tail is where an over-committed battery gets punished. **Business consequence:** the spread's width — not its mean — sets the arbitrage revenue ceiling, and its two-sidedness is why dispatch must be driven by a distributional forecast rather than a single expected price.

![Scarcity calendar](figures/fig18_scarcity_calendar.png)

Only **22 of 181 days** produced an hour above the P99 scarcity threshold ($171/MWh), and they arrive in visible bursts — early April and early May clusters — separated by long calm stretches. Scarcity is regime-driven: weather, outages, and net-load stress land together. **Business consequence:** a model trained in a calm regime can silently lose tail sensitivity, which is exactly what the Phase 6 drift monitor is built to catch before it costs money; and an operator should evaluate this asset in paydays per season, not average daily revenue.

## 2. What the model is allowed to know *(Phase 2, pending)*

<!-- fig19_feature_target_relationships -->

## 3. Forecast quality — especially the tails *(Phase 3, pending)*

<!-- fig01_forecast_fan, fig02_calibration, fig03_shap_beeswarm, fig04_shap_bar, fig05_scarcity_zoom, fig14_model_comparison, fig20_error_by_regime -->

## 4. From forecast to dispatch *(Phase 4, pending)*

<!-- fig06_dispatch_day -->

## 5. The money story *(Phase 5, pending)*

<!-- fig07_cumulative_revenue (headline), fig08_revenue_capture_bar, fig09_revenue_stack, fig10_cost_of_imperfection, fig21_soc_heatmap, fig22_reserve_allocation, fig23_monthly_revenue -->

## 6. Proof it is governed *(Phase 6, pending)*

<!-- fig11_drift_monitor, fig12_gate_status, fig13_architecture, fig24_audit_trail_flow -->
