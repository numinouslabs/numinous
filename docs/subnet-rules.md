# Subnet Rules

---

## Overview

This document defines the operational rules, constraints, and scoring mechanisms for the Numinous forecasting subnet. All miners must understand and follow these rules to participate successfully.

The key rules are the following (they will be repeated in context below):
- **Your agent re-forecasts every live event, every interval** — and carries [memory](#memory) between runs
- **You are scored against the market price, not against the outcome alone.** Matching the market scores exactly 0; beating it scores negative
- **Missing forecasts are never imputed or retried** — they cost you [coverage](./scoring-system.md#re-forecasting-pool), and falling below 85% zeroes your rewards
- **The sandbox times out after 240s**
- **The total cost limit on API calls depends on each service and is paid by the miner**
- **DO NOT include dynamic timestamps or random data in prompts to make sure our caching system is hit across different validator executions**.


**Network:** Mainnet (netuid 6), Testnet (netuid 155)

For setup instructions, see [miner-setup.md](./miner-setup.md).
For system architecture, see [architecture.md](./architecture.md).


---

# Tracks

Miners can participate in multiple **tracks**. Each track is a separate competition with its own agents, predictions, and scores. A miner's identity is `(miner_uid, hotkey, track)` — meaning the same miner can have one active agent per track.

> **Only the SIGNAL track is scored.** The re-forecasting pool on the SIGNAL track is the sole recipient of daily emissions. You can still submit a MAIN-track agent and it will still execute, but it earns nothing. See [scoring-system.md](./scoring-system.md).

- **Sandbox rules per track:** Each track defines which gateway endpoints are accessible. See [`track_config.py`](../neurons/validator/sandbox/signing_proxy/track_config.py) for the per-track endpoint allowlist — the SIGNAL one is what matters, since that is where you will be competing.
- **Credentials:** Link the services your agent uses on the track you upload to. See [linking services](./miner-setup.md#linking-services).

Available tracks are defined in [`neurons/validator/models/track.py`](../neurons/validator/models/track.py).

---

# Execution Rules

## Code Activation Schedule

**Rule:** Submitted code activates daily at **00:00 UTC**.

| Action | Timing |
|--------|--------|
| Submit code | Anytime via `numi upload-agent` |
| Activation | Next 00:00 UTC |
| First execution | For the batch of events generated that day|

**Example:**
```
23:45 UTC - You submit agent v2
00:00 UTC - Backend activates agent v2
00:XX UTC - Your agent starts executing for new events
```

**Version Management:**
- Each submission creates a new version
- Only the latest submission before midnight UTC is activated
- Previous versions are deactivated automatically
- You can submit once every three days so be mindful when you do it

## Daily Re-Forecasting Rule

**Rule:** Your agent executes **once per event, per interval** — every live event is forecast again every 24h interval until its cutoff.

Every run is a genuine execution. Nothing is carried forward: if a run fails, that interval simply has no forecast for you (see [coverage](./scoring-system.md#re-forecasting-pool)).

**To carry state between runs, use memory — not new code.** Code submissions still only activate at 00:00 UTC.

**How It Works:**

1. Event is admitted to the live cohort
2. Interval 0: your agent runs → forecast `0.65`, returns updated memory
3. Interval 1: your agent runs again, receiving that memory back → forecast `0.71`
4. Repeats every interval until cutoff


**Intervals Example:**

```
Event: "Will X happen by Jan 20?"
Cutoff: the underlying market's own resolution date

Interval 0 (Day 1): Agent executes → 0.65
Interval 1 (Day 2): Agent executes → 0.71
...
Interval N (Day N): Agent executes → 0.68
```

This is the core of the subnet: we score the **entire probability curve**, not a single point forecast.

## Memory

Your agent carries a private memory blob from one interval to the next, scoped per event.

| | |
|---|---|
| **In** | `event_data["memory"]` — what you returned for this event last interval, or `None` on the first run |
| **Out** | the `memory` key of your returned dict — a string, handed back to you next interval |
| **Scope** | per `(miner, event)`. Memory never crosses events, and never crosses miners |
| **Limit** | 32,768 characters — longer values are truncated, not rejected |
| **Optional** | omit the key and you simply receive `None` every interval |

Memory is what turns a stale forecaster into a belief-updating one: store your last probability and the reasoning behind it, then revise when new evidence arrives.

See [`memory_example.py`](../neurons/miner/agents/memory_example.py) for a worked example.


## Resource Limits

| Resource | Limit | Consequence if Exceeded |
|----------|-------|-------------------------|
| **Execution Timeout** | 240 seconds | Hard kill, no prediction recorded |
| **Code Size** | 2MB | Upload rejected |
| **Cost Limit** | Depends on service (see linking) | Run exited |
| **Python Version** | 3.11+ | - |
| **Internet Access** | None | Must use signing proxy |
| **Libraries** | Only in `sandbox/requirements.txt` | Import errors |

**Timeout Handling:**
- Agent killed after 240 seconds
- No prediction recorded = missing forecast for that interval
- Missing forecasts are **not** imputed — they count against your [coverage](./scoring-system.md#re-forecasting-pool)
- Timeouts are never retried, so every blip is permanent. Test locally to avoid this!

## Prediction Clipping

All predictions are clipped to **[0.01, 0.99]**:

```
clipped_prediction = max(0.01, min(0.99, prediction))
```

---

# Events

## Where Events Come From

Events are **prediction-market questions**. The question text, the description, and the resolution all come from the underlying market — we neither write nor resolve them. That gives two things at once: a continuous price signal to score against while the question is still open, and objective resolution when it closes.

## How Markets Are Selected

Markets are selected by **jump rate** — how often and how sharply their price repricess. Each candidate's price history is fitted with a jump-diffusion model and ranked by estimated jumps per day.

Jumpy markets are where a stale forecaster is punished hardest, and where genuine belief-updating skill separates most clearly from noise.

**Gates a market must clear before admission:**

| Gate | Value |
|------|-------|
| Price band | 0.05 – 0.95 |
| Minimum volume | $10,000 |
| Time to resolution | ≥ 14 days at admission |
| Topics | Geopolitics, economy, technology |

Markets that go quiet — under a 2-cent price range over 7 days — are removed from the cohort.

## Event Volume & Lifetime

The subnet maintains a **standing cohort of live events**, refilled daily as members resolve or are removed.

Because every live member is forecast every interval, **the cohort size is your daily forecast volume.** That size is tuned over time and is expected to grow — size your agent's runtime and API budget against the cohort you are actually served, rather than assuming a fixed count.

**Events have no fixed length.** Cutoff is the underlying market's own resolution date — at least 14 days out when admitted, and often considerably longer.

## Event Lifecycle

```
Market selected by jump rate
    ↓
Admitted to the live cohort (≥14 days to resolution)
    ↓
Every interval: miner agents execute → forecast + memory
    ↓
Each forecast scored against the market price 7 days later
    ↓
Rolling 7-day average, gated on 14-day coverage
    ↓
Weights Set
```

---

# Scoring

Scoring lives in one place: **[scoring-system.md](./scoring-system.md)**.

It covers the difficulty-adjusted formula, why matching the market scores exactly zero, the coverage gate, the eligibility ramp and confidence penalty for new miners, and how the pool is split. For the system-level mechanics, see [architecture.md](./architecture.md#scoring).

The three things that matter while reading the rules below:

- You are scored against the **market price** at the moment you forecast, not against the outcome alone.
- Missing a forecast costs you **coverage**, not a bad score — and below 85% coverage you earn nothing.
- Your score average is taken per event over the events **you** forecast, so skipping events does not dilute it.

---

# API Access

## Signing Proxy

All external API calls are routed through the validator's signing proxy. Authentication is handled for you — your agent only supplies its `RUN_ID`.

**The rule:** your agent may only call endpoints on its track's allowlist. **Anything else returns 403.** On SIGNAL that is five prefixes — the two LLM *inference* routes (OpenAI, OpenRouter), Lightning Rod, and the two Numinous signal services. The web-search variants (`/openai/responses`, `/openrouter/chat/completions`) are not on the allowlist.

The endpoint reference, with costs and request shapes, is [gateway-guide.md](./gateway-guide.md). The authoritative allowlist is [`track_config.py`](../neurons/validator/sandbox/signing_proxy/track_config.py).

**Costs:**
- API costs are paid by the miner — **link your API accounts** (see [linking services](./miner-setup.md#linking-services))
- Re-link after each agent upload — each code version needs its own link

---

# Penalties & Failures

## Execution Failures

| Failure Type | Penalty | How to Avoid |
|--------------|---------|--------------|
| **Timeout (>240s)** | Missing forecast for that interval → counts against coverage | Optimize code, test locally, add timeouts to API calls |
| **Python Error** | Missing forecast for that interval → counts against coverage | Test with `numi test-agent`, add error handling |
| **Invalid Output** | Missing forecast for that interval → counts against coverage | Validate return format: `{"event_id": str, "prediction": float}` |
| **Out of Range** | Clipped to [0.01, 0.99] | Ensure prediction in [0.0, 1.0] before returning |
| **Oversized memory** | Silently truncated to 32,768 chars | Prune your memory blob before returning it |
| **403 from gateway** | Call rejected, no data returned | Only call endpoints on the SIGNAL allowlist |
| **429 / 503 from a provider** | Depends on your retry logic | Implement exponential backoff and a fallback forecast |

## Missing forecasts

A missing forecast is not imputed and not retried — it is simply an unfilled cell in your [coverage](./scoring-system.md#re-forecasting-pool) window. This applies whether your code failed, timed out, returned something invalid, or the miner was not yet registered for that interval.

Prefer returning a mediocre forecast over crashing. A returned `0.5` scores poorly but keeps the cell filled; an exception costs you the cell outright.


## Deregistration
- A newly registered miner is immune from deregistration until it has entered the ranking — its forecasts must first become scoreable after the 7-day horizon, and it must then accumulate three scored days. The immunity period is set to cover that window, so you are not sitting at zero weight unprotected while you wait for your first scores
- After that, the lowest-weighted miner can be deregistered when a new miner registers. Failing the coverage gate zeroes your weight, so sustained downtime is the fastest route to deregistration


---

# Wallet & Registration

## Registration Requirements

- Bittensor coldkey + hotkey pair
- Registration on subnet (netuid 6 mainnet, 155 testnet)
- TAO for registration (cost fluctuates based on demand)
- Immunity period after registration

## Hotkey Verification

Your submitted code is verified against your registered wallet before execution.

**Make sure:**
- Wallet is registered on subnet
- Upload with correct wallet/hotkey, i.e hotkey matches on-chain registration

---

# Frequently Asked Questions

**Q: Are some times better than others to register?**
Yes you'd want to register the closest possible to midnight which is the activation date.

**Q: Does my agent re-execute for every 24hs interval?**
A: Yes. Your agent runs once per event per interval, every interval until that event's cutoff. Nothing is reused or carried forward.

**Q: Can I update a prediction for an event?**
A: Yes — that is the point. Each interval is a fresh forecast, and the whole curve is scored. Use [memory](#memory) to carry your reasoning between runs.

**Q: How many events will my agent process?**
A: However many are in the live cohort — every one of them, every interval. The cohort size is tuned over time and is expected to grow, so don't build against a fixed number.

**Q: When does my submitted code become active?**
A: At the next **00:00 UTC** after submission.

**Q: If I just return the market price, what do I score?**
A: Exactly 0 — the two terms of the score cancel. Matching the market is the baseline you have to beat, not a strategy.

**Q: What happens if my agent times out?**
A: Execution is killed after 240 seconds and no forecast is recorded for that interval. It is not imputed and not retried; it counts as a missed cell against your [coverage](./scoring-system.md#re-forecasting-pool).

**Q: Can I submit multiple times per day?**
A: No, you can submit once every three days, so please ensure you really test it before uploading you code.

---

# Rules Summary Checklist

Before submitting your agent, ensure:

- ✅ Code implements `agent_main(event_data) -> {"event_id": str, "prediction": float}`
- ✅ Reads `event_data["memory"]` and handles `None` on the first interval
- ✅ Returns a `memory` string under 32,768 characters
- ✅ Execution time < 240 seconds (tested locally)
- ✅ Code size < 2MB
- ✅ Uses only libraries in `sandbox/requirements.txt`
- ✅ Returns predictions in [0.0, 1.0] range
- ✅ Never raises — returns a fallback forecast instead, to keep the coverage cell filled
- ✅ Only calls endpoints on the SIGNAL allowlist (anything else returns 403)
- ✅ Implements proper error catching (403, 503, 404, 429)
- ✅ Tested with `numi test-agent` before submission
- ✅ Wallet registered on subnet
- ✅ Submitted before midnight UTC to activate next day

---

**Next:** See [miner-setup.md](./miner-setup.md) for setup instructions and [architecture.md](./architecture.md) for system details.
