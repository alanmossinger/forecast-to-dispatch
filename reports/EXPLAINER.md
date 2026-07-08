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

### 2. Features — what the model is allowed to know

The forecast for tomorrow is issued at **9 a.m. today**, before ERCOT's day-ahead market closes. The model's inputs are only things genuinely knowable at that moment: real-time prices through *the day before yesterday*, day-ahead prices through *today* (they were published yesterday afternoon), recent system load and wind/solar output, and the calendar.

How do we know nothing from the future leaks in? We *prove* it: take one delivery day, delete from the dataset everything that was still unknown at its 9 a.m. issue moment, rebuild the model's inputs from the censored data — and check they come out identical. They do, for every feature, and this experiment runs automatically in the release gates. Why it matters: future leakage is how storage backtests quietly overstate revenue by 20% or more; this system makes that failure impossible rather than merely unlikely.

### 3. Forecast — five quantiles, not one number

Instead of predicting "tomorrow at 8 p.m. power will cost \$45", the model predicts a *range*: "there's a 5% chance it's below \$20, a 50% chance below \$45, a 5% chance above \$400". That top number — the P95 — is the money number: when it jumps, the system knows a scarcity evening may be coming and holds the battery's charge.

Two honesty mechanisms guard the forecast. First, the stated ranges are *calibrated*: when the model says "90% confident," reality lands inside 86% of the time (measured on months the model never saw — and repaired with a statistical technique called conformal calibration after the raw model proved overconfident). Second, every forecast is *explainable*: a standard technique (SHAP) decomposes each prediction into named drivers — recent prices, the market's own day-ahead expectation, electricity demand net of wind and solar — so a human reviewer can interrogate any forecast in plain market terms.

We also trained a fashionable deep-learning model (an LSTM) behind the same interface and scored it identically. It lost on every metric, so the simpler, explainable model ships — a decision made on evidence, and documented, which is itself part of the governance story.

### 4. Dispatch — the forecast becomes a plan

An optimizer turns the price forecast into a 24-hour operating plan: buy energy in the cheap hours, sell it in the expensive ones, and in between rent the battery out as *standby capacity* — the grid pays for megawatts held ready (reserves), often more reliably than the buy-low/sell-high trade itself.

The optimizer is bound by the battery's physics, written as hard constraints it cannot violate: energy is conserved (with ~8% round-trip loss), the tank can't overfill, the inverter has a power rating, any megawatt promised as reserve must be backed by real headroom *and* real stored energy, and every cycle pays a wear-and-tear cost. The system checks every schedule against these rules before releasing it, and automated tests prove the checks work. One subtlety: when power prices go negative (windy Texas nights), a naive optimizer tries to waste energy by charging and discharging at once — the system detects this and switches to a stricter solve that forbids it.

### 5. Backtest — *(Phase 5, pending)*
Decide on the forecast, settle on reality: the mechanism that keeps the revenue claim honest.

### 6. Governance — *(Phase 6, pending)*
The model card, risk register, drift monitor, audit trail, human approval gate, and rollback path — and the single command that blocks an ungoverned model from shipping.

### 7. Serving — *(Phase 7, pending)*
The API an operator would call, with the audit log and approval gate wired into every request.
