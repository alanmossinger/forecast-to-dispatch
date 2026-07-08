# How a price forecast becomes a bid becomes revenue — and where the guardrails sit

*A plain-language walkthrough for the non-coder reader. Updated as each module lands.*

---

## The one-paragraph version

A grid-scale battery makes money by buying power when it is cheap and selling it — as energy or as standby reserves — when it is expensive. Prices in ERCOT swing from below \$0 to thousands of dollars per MWh, and most of a battery's annual revenue arrives in a handful of scarcity hours. This system (1) forecasts a **range** of possible prices for every hour of tomorrow, (2) computes the most profitable battery schedule against that forecast, and (3) refuses to act unless a set of governance checks — and a human — sign off. Every decision is logged permanently, and the reported revenue is settled at the prices that *actually happened*, never at the forecast.

## Stage by stage

### 1. Data — where the prices come from

Three public ERCOT price streams are pulled from the ISO's historical archives (via the open-source `gridstatus` library) and aligned on one hourly clock at the Houston trading hub:

- **Day-ahead prices** — what power for each hour of tomorrow sold for in yesterday's auction.
- **Real-time prices** — what power actually cost in each 15-minute interval (averaged to hourly). This is where scarcity spikes appear: the same megawatt-hour that averages \$27 can cost \$3,000 for an hour on a stressed evening.
- **Ancillary-service clearing prices** — what the grid pays a resource just to *stand ready* (Regulation Up/Down, Responsive Reserve, Non-Spin, ECRS). A battery can earn these while holding its charge.

Two honesty rules are enforced in code: the assembled dataset **fails loudly if any hour is missing** (a silent gap would corrupt every revenue number downstream), and a small **offline sample ships with the repository** so anyone — including the automated CI gates — can rerun the entire pipeline with no network access and get the same numbers.

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
