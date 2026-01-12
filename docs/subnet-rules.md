# Subnet Rules

---

## Overview

This document defines the operational rules, constraints, and scoring mechanisms for the Numinous forecasting subnet. All miners must understand and follow these rules to participate successfully.

The key rules are the following (they will be repeated in context below):
- **The sandbox times out after 150s**
- **The total cost limit on API calls is $0.02**
- **DO NOT include dynamic timestamps or random data in prompts to make sure our caching system is hit across different validator executions**.


**Network:** Mainnet (netuid 6), Testnet (netuid 155)

For setup instructions, see [miner-setup.md](./miner-setup.md).
For system architecture, see [architecture.md](./architecture.md).


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

## One Prediction Per Event Rule

**Rule:** Your agent executes **once per event** and produces a single prediction that is used for the entire event lifecycle.

**To change strategy, submit new agent code (active next day for new events only).**


**How It Works:**

1. Event created (e.g., 2025-01-15 10:00 UTC)
2. Your agent executes → produces prediction (e.g., 0.65)
3. Prediction is reused for all 24hs intervals until cutoff (spot scoring). Your agent does NOT re-execute for that event


**Intervals Example:**

```
Event: "Will X happen by Jan 20?"
Cutoff: 2025-01-20 23:59 UTC

Interval 0 (Day 1): Agent executes → 0.65
Interval 1 (Day 2): Prediction reused → 0.65
...
Interval N (Day N): Prediction reused → 0.65
```

We do this currently for **efficient ressource usage**. This will change in the future since there is clear benefit in having multiple forecasting schedules. 


## Resource Limits

| Resource | Limit | Consequence if Exceeded |
|----------|-------|-------------------------|
| **Execution Timeout** | 150 seconds | Hard kill, no prediction recorded |
| **Code Size** | 2MB | Upload rejected |
| **Cost Limit** | $0.02 per run | Run exited |
| **Python Version** | 3.11+ | - |
| **Internet Access** | None | Must use signing proxy |
| **Libraries** | Only in `sandbox/requirements.txt` | Import errors |

**Timeout Handling:**
- Agent killed after 150 seconds
- No prediction recorded = missing prediction
- Imputed prediction = 0.5
- Test locally to avoid this!

---

# Event Generation

## Event Volume

**Approximately 100 events generated daily** across all event types.

Events are generated throughout the day (not uniformly distributed).

**All events are currently 3 days events.** This is to align with the immunity period and ensure as much as possible that at each scoring cadence all the miners are scored on the same events.

## Event Types

### 1. LLM Generated Events

Geopolitical and political events from news triggers:

**Examples:**
- "Will the US confirm arms delivery to Ukraine by March 20, 2025?"
- "Will the Trump administration exclude tariffs on cars by April 2, 2025?"

**Sources:** Polymarket triggers

### 2. Polymarket

Subset of Polymarket markets where the midprice is within 0.2-0.8 when received.

## Event Lifecycle

```
Event Created
    ↓
Miner Agents Execute (once per event)
    ↓
Predictions Submitted
    ↓
Event Cutoff Reached (24h before resolution typically)
    ↓
Event Resolved (outcome = 0 or 1)
    ↓
Brier Scoring Calculated
    ↓
Metagraph Scores Updated
    ↓
Weights Set
```

---

# Scoring System

## Brier Score

The scoring system uses the **Brier score** to measure prediction accuracy.

**Formula:**
```
Brier Score = (prediction - outcome)²

Where:
- prediction: Your probability forecast (0.0 to 1.0)
- outcome: 1 if the event happened, 0 if the event did not (all events are binary)
```
**Lower is better** (0.0 = perfect, 1.0 = worst)

If your agent does not submit a prediction, it gets imputed 0.5 as your prediction.

The miners are scored by their average Brier score over the last 101 events. The miner with the lowest average Brier score gets all the rewards.


## Prediction Clipping

All predictions are clipped to **[0.01, 0.99]**:

```
clipped_prediction = max(0.01, min(0.99, prediction))
```

---

# API Access

## Signing Proxy

All external API calls routed through validator's signing proxy.

**Available Endpoints:**
- `/api/gateway/chutes` - Chutes AI (open-source LLM API)
- `/api/gateway/desearch` - Desearch AI (Web/Twitter search)

**Authentication & Costs:**
- Authentication handled automatically by signing proxy
- **Link your API accounts to access higher budgets** (see [miner-setup.md](./miner-setup.md#linking-services))
  - `numi services link chutes` - Link Chutes API key
  - `numi services link desearch` - Link Desearch API key
- Re-link after each agent upload

## Chutes AI - Open Source LLMs

Chutes provides access to **open-source language models** (e.g., DeepSeek, Qwen, Mistral).

**Note:** Chutes does NOT provide OpenAI models (GPT-4, GPT-3.5, etc.). Only open-source models are available.

### Hot vs Cold Models

**Hot models** (active instances):
- Ready immediately
- Return results fast
- Recommended for production

**Cold models** (no active instances):
- Return `503 Service Unavailable`
- Must wait for instance to warm up
- Can cause agent timeout

**Check Model Availability:**
Visit https://chutes.ai/app to see which models are "hot" (have active instances) before submitting your agent.

---

# Penalties & Failures

## Execution Failures

| Failure Type | Penalty | How to Avoid |
|--------------|---------|--------------|
| **Timeout (>150s)** | No prediction, imputed 0.5 → Brier score = 0.25 | Optimize code, test locally, add timeouts to API calls |
| **Python Error** | No prediction, imputed 0.5 → Brier score = 0.25 | Test with `numi test-agent`, add error handling |
| **Invalid Output** | No prediction, imputed 0.5 → Brier score = 0.25 | Validate return format: `{"event_id": str, "prediction": float}` |
| **Out of Range** | Clipped to [0.01, 0.99] | Ensure prediction in [0.0, 1.0] before returning |
| **503 from Chutes** | Depends on retry logic | Use hot models, implement backoff, have fallback |
| **404 from Chutes** | Model not found | Check https://chutes.ai/app for available models |
| **429 from Chutes** | Rate limit exceeded | Implement exponential backoff, reduce API calls |

## Missing prediction
The miner is imputed a prediction of 0.5 in all cases of missing prediction: 
- for intervals before registration when registering a miner
- if the code does not return a prediction or fails


## Deregistration
- A miner has an initial immunity period of approximately 3 days where it can't be de-registered
- After the immunity period the miner with the lowest Brier score can be deregistered if new miner registers


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
A: No. At the moment your agent executes **once per event**. The same prediction is automatically reused for all intervals until cutoff. 

**Q: Can I update a prediction for an event?**
A: No. Once submitted, predictions are final. To change strategy, submit new agent code (active next day for new events only).

**Q: How many events will my agent process?**
A: Approximately 100 events per day across all event types.

**Q: When does my submitted code become active?**
A: At the next **00:00 UTC** after submission.

**Q: What happens if my agent times out?**
A: Execution is killed after 150 seconds. No prediction is recorded. You get imputed prediction of 0.5, resulting in Brier score of 0.25.

**Q: What if I get a 503 error from Chutes?**
A: You requested a cold model. Use hot models (check https://chutes.ai/app) and implement exponential backoff retry logic with a fallback.

**Q: Can I submit multiple times per day?**
A: No, you can submit once every three days, so please ensure you really test it before uploading you code.

---

# Rules Summary Checklist

Before submitting your agent, ensure:

- ✅ Code implements `agent_main(event_data) -> {"event_id": str, "prediction": float}`
- ✅ Execution time < 150 seconds (tested locally)
- ✅ Code size < 2MB
- ✅ Uses only libraries in `sandbox/requirements.txt`
- ✅ Returns predictions in [0.0, 1.0] range
- ✅ Has error handling and fallback logic
- ✅ Uses hot Chutes models with retry logic
- ✅ Implements proper error catching (503, 404, 429)
- ✅ Tested with `numi test-agent` before submission
- ✅ Wallet registered on subnet
- ✅ Submitted before midnight UTC to activate next day

---

**Next:** See [miner-setup.md](./miner-setup.md) for setup instructions and [architecture.md](./architecture.md) for system details.
