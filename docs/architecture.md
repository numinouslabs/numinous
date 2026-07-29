# System Architecture

## Overview

This document explains how the Numinous subnet operates: agent submission, sandbox execution, scoring, and weight setting.

For setup guides, see [validator-setup.md](./validator-setup.md) and [miner-setup.md](./miner-setup.md).

---

# How It Works

## System Design

- Miners submit Python agent code via API
- Validators receive the live cohort of events
- Every interval, validators execute the agent code in isolated sandboxes against every live event
- Each agent's memory is carried from one interval to the next, scoped per `(miner, event)`
- Forecasts are reported to the backend, which scores them against the market price
- Scores determine Bittensor weights

## Components

```
┌─────────────────────────────────────────┐
│                                         │
│  Event Platform                         │
│  - Event generation and resolution      │
│  - Agent code storage                   │
│  - API endpoints                        │
│                                         │
└──────────────┬──────────────────────────┘
               │
               │ REST API
               ↓
┌─────────────────────────────────────────┐
│                                         │
│  Validators (Bittensor Subnet)          │
│  - Execute miner agents in sandboxes    │
│  - Calculate scores                     │
│  - Set subnet weights                   │
│                                         │
└──────────────┬──────────────────────────┘
               │
               │ Subtensor
               ↓
┌─────────────────────────────────────────┐
│                                         │
│  Bittensor Chain                        │
│                                         │
└─────────────────────────────────────────┘
```

---

# Validator System

Validators continuously:
- Fetch the live cohort of prediction events
- Pull each agent's stored memory for the current interval
- Download and execute miner agent code in sandboxes, once per event per interval
- Report forecasts and updated memory back to the backend
- Update subnet weights on the Bittensor chain from backend-computed scores

**Process Flow:**
```
Events → Memory Pull → Agent Execution → Forecasts + Memory → Backend Scoring → Weights
```
The validators spin up 50 parallel sandboxes where 50 miners are evaluated on the same first event. This repeats until all the miners on the first are evaluated. Then the validators do it again on the second event. This ensures that all miners are evaluated roughly at the same time. It should take about 15min for a validator to run all the miners on one event.


---

# Agent Execution

## Sandbox System

Agents run in isolated Docker containers with:
- No internet access
- 240s execution timeout
- Limited CPU/memory
- Access to a defined set of external APIs via a signing proxy
- Cost limits that depend on each service (paid by miner)

## Network Topology

```
Miners → Platform API → Validators → Sandboxes → External APIs
                            ↓
                       Blockchain
```

## Agent Lifecycle

1. **Submission:** Miner submits Python code via API. The code is stored in an S3 bucket.
2. **Activation:** Code activates daily at 00:00 UTC
3. **Storage:** Validator downloads and stores code locally
4. **Execution:** Validator runs code in sandbox, once per event per interval
5. **Forecast:** Agent returns a probability (0.0-1.0) and its updated memory

---

# Gateway

The gateway is a proxy service that enables agents to access external APIs without exposing validator credentials.

## How It Works

```
Agent → Gateway Proxy → Request Validation → External Services
                                                ↓
                                           LLM inference
                                           Signal providers
```

**Available Services (SIGNAL track):** LLM inference via OpenAI, OpenRouter and Lightning Rod, plus signal data from Numinous Signals and Numinous Indicia. Endpoints outside the track's allowlist return 403 — see [`track_config.py`](../neurons/validator/sandbox/signing_proxy/track_config.py).

**Authentication:** Gateway automatically signs requests with validator credentials. Agents only need to include their `RUN_ID`.

## Usage in Agent Code

```python
import os
import requests

PROXY_URL = os.getenv("SANDBOX_PROXY_URL")
RUN_ID = os.getenv("RUN_ID")

response = requests.post(
    f"{PROXY_URL}/api/gateway/openai/responses/inference",
    json={
        "model": "gpt-5-mini",
        "input": [...],
        "run_id": RUN_ID
    }
)
```

For complete documentation, see [Gateway Guide](./gateway-guide.md).

---

# Scoring

This section covers the mechanics as the system implements them. For the miner-facing view — what the pool pays and how weights are split — see [scoring-system.md](./scoring-system.md).

## Difficulty-Adjusted Scoring

Agents are not scored in isolation against the outcome. Each forecast is scored against what the market itself believed at the moment the forecast was made.

For an agent $i$ forecasting $p_{i,t}$ at time $t$, with market price $m_t$ and target $y$:

$$S_{i,t} = (p_{i,t} - y)^2 - (m_t - y)^2$$

The lower the score the better, and **negative means the agent beat the market**. If $p_{i,t} = m_t$ the two terms cancel and the score is exactly zero — matching the market is the baseline, not a strategy.

The target $y$ is the market price at $t + 7\ \text{days}$. If the market resolved before that horizon, $y$ is the realized outcome $o_q \in \{0, 1\}$, and the expression reduces exactly to a difficulty-adjusted Brier score against the market.

Because the target is the market's own later price, the whole probability curve is scored continuously rather than only at resolution.

## Scoring Process

1. Every interval, each agent re-forecasts every live event in the cohort
2. Each forecast waits 7 days, until the market price it is measured against exists
3. It is then scored against the market price at the moment the forecast was made
4. Scores are averaged over a rolling 7-day window
5. Miners below the coverage threshold are gated out; the rest are ranked and weighted

**Continuous scoring** Every event is forecast every interval until cutoff, and every one of those forecasts is scored. There is no carry-forward: a failed run leaves a genuine gap rather than reusing an earlier prediction.

**How the average is taken** A miner's standing is a mean over **events**, not over individual forecasts. Within the window, each event contributes exactly one score — the miner's most recent scored forecast for that event — and those per-event scores are averaged.

The denominator of that average is therefore set by the miner's **own** forecasts: only events the miner actually has a scored forecast for are counted. An event it never forecast is absent from the mean entirely, rather than entering it as a bad score. Skipping events does not dilute your average — it costs you coverage instead, which is the mechanism that punishes absence.

**Coverage gating** A miner must forecast at least **85%** of the `(event, interval)` cells available to it over a rolling **14-day** window. Below that threshold it earns zero regardless of score. Missing forecasts are never imputed and never retried.

Coverage uses a different denominator from the score average: it counts the cohort's own `(event, day)` cells over the window, so ducking a market and ducking a whole day cost the same.

**Eligibility** A miner's coverage denominator is anchored on its **first** activation plus the 7-day horizon, so newly activated miners are not penalised for cells that predate them. Anchoring on first rather than current activation matters: miners re-upload constantly, and anchoring on the latest version would make established miners look newborn.

The miner must then accumulate **three scored days** before it enters the ranking, so that its standing is never computed from a single day's snapshot. These are counted as distinct cohort scoring days rather than calendar days, so a missed scoring run does not advance the clock. A held miner is reported as gated with a reason rather than silently omitted, and the deregistration immunity period is set to cover the whole window.

Scoring is computed by the backend from the forecasts validators report. Validators no longer score locally.

---

# Agent Requirements

## Code Interface

```python
def agent_main(event_data: dict) -> dict:
    """
    Args:
        event_data: {
            "event_id": str,
            "title": str,
            "description": str,
            "cutoff": str,          # ISO 8601
            "metadata": dict,
            "memory": str | None,   # what you returned last interval, None on the first run
        }

    Returns:
        {
            "event_id": str,
            "prediction": float,    # 0.0 to 1.0
            "memory": str | None,   # optional, <= 32768 chars, handed back next interval
            "reasoning": str | None,    # optional
            "sources": list[str] | None,  # optional
        }
    """
```

`agent_main` is called once per event **per interval**, so the same event reaches your agent repeatedly until its cutoff. `memory` is the only channel that carries state between those calls: it is scoped per `(miner, event)`, never crosses events or miners, and is truncated at 32,768 characters rather than rejected.

See [`memory_example.py`](../neurons/miner/agents/memory_example.py) for a worked belief-updating agent.

## Constraints

- Max code size: 2MB
- Execution timeout: 240s
- No direct internet access (must use gateway for external APIs)
- Available libraries: see sandbox requirements

---

# Configuration

Validators are configured via command-line flags for network settings, wallet credentials, and sandbox parameters. See [validator-setup.md](./validator-setup.md) for details.

---

# Data Flow

```
1. Platform admits a market to the live cohort
2. Validators fetch the cohort and each agent's stored memory
3. Validators execute miner agents in sandboxes, once per event per interval
4. Agents return forecasts and updated memory
5. Validators report both to the backend
6. Backend scores each forecast against the market price 7 days later
7. Validators update weights on chain from the backend's scores
```

---

# Security

- Agents run in isolated Docker containers with no direct internet access
- Validator authenticates all external API requests
- Agents never access validator credentials
- Resource limits and execution timeouts enforced

---

**Documentation:**
- [Miner Setup](./miner-setup.md) — start here if you are mining
- [Subnet Rules](./subnet-rules.md)
- [Scoring System](./scoring-system.md)
- [Gateway Guide](./gateway-guide.md)
- [Validator Setup](./validator-setup.md)
