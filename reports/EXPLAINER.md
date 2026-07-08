# How a price forecast becomes a bid becomes revenue — and where the guardrails sit

*A plain-language walkthrough for the non-coder reader. Updated as each module lands.*

---

## The one-paragraph version

A grid-scale battery makes money by buying power when it is cheap and selling it — as energy or as standby reserves — when it is expensive. Prices in ERCOT swing from below \$0 to thousands of dollars per MWh, and most of a battery's annual revenue arrives in a handful of scarcity hours. This system (1) forecasts a **range** of possible prices for every hour of tomorrow, (2) computes the most profitable battery schedule against that forecast, and (3) refuses to act unless a set of governance checks — and a human — sign off. Every decision is logged permanently, and the reported revenue is settled at the prices that *actually happened*, never at the forecast.

## Stage by stage

### 1. Data — *(Phase 1, pending)*
Where the prices come from and why we align day-ahead, real-time, and ancillary-service prices on one hourly clock.

### 2. Features — *(Phase 2, pending)*
What the model is allowed to know at forecast time, and how we prove nothing from the future leaks in.

### 3. Forecast — *(Phase 3, pending)*
Why we forecast five quantiles instead of one number, and how we measure honesty about uncertainty.

### 4. Dispatch — *(Phase 4, pending)*
How the optimizer turns a price forecast into an hourly charge/discharge/reserve schedule without ever breaking the battery's physics.

### 5. Backtest — *(Phase 5, pending)*
Decide on the forecast, settle on reality: the mechanism that keeps the revenue claim honest.

### 6. Governance — *(Phase 6, pending)*
The model card, risk register, drift monitor, audit trail, human approval gate, and rollback path — and the single command that blocks an ungoverned model from shipping.

### 7. Serving — *(Phase 7, pending)*
The API an operator would call, with the audit log and approval gate wired into every request.
