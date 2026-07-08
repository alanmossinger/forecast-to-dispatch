# Risk register — Forecast-to-Dispatch

*Generated from `governance/risk_register.yaml` (source of truth). Schema-validated in CI; a malformed or empty register fails the release gate.*

| ID | Risk | Likelihood | Impact | Status | Owner |
|---|---|---|---|---|---|
| R2 | Market regime drift (weather, fuel, fleet change) | high | high | mitigated-monitored | model-owner |
| R1 | Forecast miss during a scarcity event | medium | high | mitigated-monitored | model-owner |
| R6 | Future leakage in features | medium | high | mitigated | model-owner |
| R7 | Overconfident uncertainty intervals | medium | high | mitigated-monitored | model-owner |
| R3 | State-of-charge or reserve constraint violation | low | high | mitigated | dispatch-owner |
| R4 | Market rule change (RTC+B evolution, product redefinition) | medium | medium | accepted-with-controls | market-owner |
| R5 | Data outage or archive expiry | medium | medium | mitigated | data-owner |
| R8 | Unapproved or tampered dispatch reaching deployment | low | high | mitigated | operations-owner |
| R9 | Bad model version in serving | medium | medium | mitigated | operations-owner |
| R10 | Optimizer failure or pathological solution | low | medium | mitigated | dispatch-owner |

## R2 — Market regime drift (weather, fuel, fleet change)

**Likelihood:** high · **Impact:** high · **Status:** mitigated-monitored · **Owner:** model-owner

The joint distribution of prices and net load shifts away from the training window; model skill decays silently.

**Mitigation:** PSI + KS drift monitor on key features and the prediction distribution with alert thresholds (fig11); governance.check fails the release gate on corroborated drift; retraining playbook = rerun forecast stage.

## R1 — Forecast miss during a scarcity event

**Likelihood:** medium · **Impact:** high · **Status:** mitigated-monitored · **Owner:** model-owner

The P95 fails to flag a spike evening; the battery enters the window under-charged or committed to reserves, missing the highest-value hours.

**Mitigation:** Scarcity recall tracked as a first-class metric (fig05, fig14); conformal calibration keeps intervals honest; fig10 quantifies the realized cost; drift monitor (R2) watches for regime change that degrades tail vision.

## R6 — Future leakage in features

**Likelihood:** medium · **Impact:** high · **Status:** mitigated · **Owner:** model-owner

A feature quietly encodes post-issue-time information, inflating backtest revenue that production can never realize.

**Mitigation:** Declared minimum legal lags (FEATURE_SPEC) enforced at build time; the truncation experiment (blind the future, features must be identical) runs in CI; backtest capture sanity test flags near-100% capture as suspect.

## R7 — Overconfident uncertainty intervals

**Likelihood:** medium · **Impact:** high · **Status:** mitigated-monitored · **Owner:** model-owner

The stated 90% interval covers less than promised; dispatch under-positions for tails and risk reporting is fiction.

**Mitigation:** Width-scaled conformal calibration with finite-sample guarantee (raw model was at 68% coverage - repaired to 86%); coverage tolerance is a release gate in governance.check.

## R3 — State-of-charge or reserve constraint violation

**Likelihood:** low · **Impact:** high · **Status:** mitigated · **Owner:** dispatch-owner

A schedule that overpromises energy or reserves would fail in the market and can incur ISO penalties.

**Mitigation:** Constraints enforced in the optimizer AND re-asserted post-solve (_assert_physical refuses invalid schedules); property tests cover SOC dynamics, headroom, energy-backing, and charge/discharge exclusivity.

## R4 — Market rule change (RTC+B evolution, product redefinition)

**Likelihood:** medium · **Impact:** medium · **Status:** accepted-with-controls · **Owner:** market-owner

ERCOT protocol changes alter product definitions or co-optimization rules; the dispatch model silently optimizes a market that no longer exists.

**Mitigation:** Product set and market parameters are config, not code; risk register review is part of the release checklist; operational control - subscribe to ERCOT market notices (production).

## R5 — Data outage or archive expiry

**Likelihood:** medium · **Impact:** medium · **Status:** mitigated · **Owner:** data-owner

ERCOT MIS reports expire (observed during build - daily reports for 2024 were already gone); a silent gap would corrupt features and settlement.

**Mitigation:** Ingest validates hourly continuity and fails loudly on any gap; per-year raw caching; committed offline sample keeps CI independent of the source; historical-archive fallbacks documented in ingest.py.

## R8 — Unapproved or tampered dispatch reaching deployment

**Likelihood:** low · **Impact:** high · **Status:** mitigated · **Owner:** operations-owner

An autonomous schedule ships without human review, or an approved schedule is modified after approval.

**Mitigation:** HITL gate - approval is bound to the SHA-256 of the schedule content and logged; API refuses deployment without it; audit log is append-only with a verified hash chain (tamper-evident).

## R9 — Bad model version in serving

**Likelihood:** medium · **Impact:** medium · **Status:** mitigated · **Owner:** operations-owner

A degraded retrain reaches the API and dispatches on worse forecasts.

**Mitigation:** Versioned registry with metrics.json per model; release gates check the CURRENT model's metrics; one-command rollback re-points to the previous version and is covered by a test.

## R10 — Optimizer failure or pathological solution

**Likelihood:** low · **Impact:** medium · **Status:** mitigated · **Owner:** dispatch-owner

Solver returns infeasible/unbounded status, or an LP artifact (simultaneous charge-discharge under negative prices) slips through.

**Mitigation:** Non-optimal solver status raises; automatic MILP escalation on detected overlap; post-solve physical assertions; negative-price stress test in CI.
