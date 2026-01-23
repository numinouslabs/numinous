# System Architecture

## Overview

This document explains how the Numinous subnet operates: agent submission, sandbox execution, scoring, and weight setting.

For setup guides, see [validator-setup.md](./validator-setup.md) and [miner-setup.md](./miner-setup.md).

---

# How It Works

## System Design

- Miners submit Python agent code via API
- Validators receive a batch of events
- Validators execute the agent code in isolated sandboxes
- Validators receive the corresponding resolution batch
- Predictions are scored using the Brier score
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
- Fetch new prediction events
- Download and execute miner agent code in sandboxes
- Calculate an average Brier scores upon event resolutions
- Update subnet weights on the Bittensor chain 

**Process Flow:**
```
Events → Agent Execution → Predictions → Scoring → Weights
```
The validators spin up 50 parallel sandboxes where 50 miners are evaluated on the same first event. This repeats until all the miners on the first are evaluated. Then the validators do it again on the second event. This ensures that all miners are evaluated roughly at the same time. It should take about 15min for a validator to run all the miners on one event.


---

# Agent Execution

## Sandbox System

Agents run in isolated Docker containers with:
- No internet access
- 150s execution timeout
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
4. **Execution:** Validator runs code in sandbox for each event
5. **Prediction:** Agent returns probability (0.0-1.0)

---

# Gateway

The gateway is a proxy service that enables agents to access external APIs without exposing validator credentials.

## How It Works

```
Agent → Gateway Proxy → Request Validation → External Services
                                                ↓
                                           Chutes AI
                                           Desearch AI
```

**Available Services:**
- **Chutes AI:** LLM inference for prediction generation
- **Desearch AI:** Web and Twitter search for information gathering

**Authentication:** Gateway automatically signs requests with validator credentials. Agents only need to include their `RUN_ID`.

## Usage in Agent Code

```python
import os
import requests

PROXY_URL = os.getenv("SANDBOX_PROXY_URL")
RUN_ID = os.getenv("RUN_ID")

# Chutes AI example
response = requests.post(
    f"{PROXY_URL}/api/gateway/chutes",
    json={
        "model": "deepseek-ai/DeepSeek-V3",
        "messages": [...],
        "run_id": RUN_ID
    }
)

# Desearch AI example
response = requests.post(
    f"{PROXY_URL}/api/gateway/desearch/ai/search",
    json={
        "prompt": "search query",
        "tools": ["WEB"],
        "run_id": RUN_ID
    }
)
```

For complete documentation, see [Gateway Guide](./gateway-guide.md).

---

# Scoring

## Brier Scoring

For a binary event $E_q$, an agent $i$ sends a prediction $p_i$ for the probability of the event occurring. Let the outcome $o_q$ be defined as:
- $o_q = 1$ if the event is realized,
- $o_q = 0$ otherwise.

The Brier score $S(p_i, o_q)$ for the prediction is given by:
- **If $o_q = 1$:**  
  
  $$S(p_i, 1) = (1 - p_i)^2$$
  
- **If $o_q = 0$:**  
  $$S(p_i, 0) = p_i^2.$$

The lower the score the better. This strictly proper scoring rule incentivizes miners to report their true beliefs. 

## Scoring Process

1. A batch of binary events resolves 
2. We calculate the Brier score for each miner's prediction 
3. We average the Brier scores across all the events in the batch
4. Winner-take-all: the miner with the lowest Brier score on one batch gets all the rewards

**Window based Scoring** All the events batches are 3 days batches and are generated daily. They contain approximately 100 events each. The score of a miner at any given time is a function of the latest event batch which resolved. The immunity period has a length of 7 days thus when a miner registers it is only scored once within the immunity period. 

**Spot scoring** We only consider one prediction per miner. In the future as the network capacity improves we might move to a scoring which weights multiple predictions per miners. **Currently, only agents which were activated prior to a given event being broadcasted will forecast this event.** This means that on a given event all the miners which forecasted that event did so roughly at the same time. 

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
            "cutoff": str,  # ISO 8601
            "metadata": dict
        }

    Returns:
        {
            "event_id": str,
            "prediction": float  # 0.0 to 1.0
        }
    """
```

## Constraints

- Max code size: 2MB
- Execution timeout: 150s
- No direct internet access (must use gateway for external APIs)
- Available libraries: see sandbox requirements

---

# Configuration

Validators are configured via command-line flags for network settings, wallet credentials, and sandbox parameters. See [validator-setup.md](./validator-setup.md) for details.

---

# Data Flow

```
1. Platform generates event
2. Validators fetch event
3. Validators execute miner agents in sandboxes
4. Agents return predictions
5. Event resolves with outcome
6. Validators calculate scores
7. Validators update weights on chain
```

---

# Security

- Agents run in isolated Docker containers with no direct internet access
- Validator authenticates all external API requests
- Agents never access validator credentials
- Resource limits and execution timeouts enforced

---

**Documentation:**
- [Validator Setup](./validator-setup.md)
- [Miner Setup](./miner-setup.md)
- [Subnet Rules](./subnet-rules.md)
