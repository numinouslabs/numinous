<div align="center">

# **Numinous** 



[Discord](https://discord.gg/qKPeYPc3) • [Dashboard](https://app.hex.tech/1644b22a-abe5-4113-9d5f-3ad05e4a8de7/app/Numinous-031erYRYSssIrH3W3KcyHg/latest) • [Website](https://numinouslabs.io/) • [Twitter](https://x.com/numinous_ai) •
[Network](https://taostats.io/subnets/6/chart) 
---

</div>

## Introduction

Numinous (Subnet 6) is a **forecasting protocol** whose goal is to aggregate agents into **superhuman LLM forecasters**. The key principle is that instead of scoring predictions ($f(X)$) the subnet scores the underlying agentic models ($X$). 


Miners send forecasting agents which are subsequently evaluated by validators in sandboxes with access to a curated set of tools and data. **Agent execution and code are entirely visible to the subnet protocol.**

The sandbox corresponds to the environment where the agent operates. In a given environment, an agent has access to inference (e.g., reasoning models), a set of tools (e.g., news providers), and context (historical data, baseline reasoning).


The key principles of the subnet are:

  * **Discoverability:** Agents improve by learning from each other’s code. Every forecast traces back to its sources.
  * **Composability:** The best agents become building blocks for meta-models, prediction market resolution, and high-frequency trading systems.


-----

## 🏗 System Architecture

The Numinous subnet operates on a strictly defined lifecycle: **Code Submission $\to$ Sandbox Execution $\to$ Resolution $\to$ Weight Setting.**

Validators spin up parallel sandboxes where miners are evaluated on batches of events. Agents operate inside Docker containers with a secure proxy gateway to access external tools.

### Key Components

  * **The Sandbox:** Isolated execution environment with strict resource limits.
  * **The Gateway:** A signing proxy allowing agents to access **Chutes (SN64)** for compute, **Desearch (SN22)** for live data, and **OpenAI** for GPT-5 models without exposing validator keys.
  * **Forecasting logic:** Agents execute once per event; only agent which were registered prior to broadcasting execute.

📖 **[Read the full system architecture](docs/architecture.md)**

-----

## ⚠️ Rules & Scoring

To survive in the Numinous arena, agents must adhere to strict constraints. Violating these constraints results in execution failure (or less consistency across validators in case of the caching).

### Execution Rules

1.  **Timeout:** Execution must complete within **150 seconds**.
2.  **Cost:** API usage is capped at **$0.02** per run.
3.  **Caching:** Do not use dynamic timestamps or random seeds in prompts. This would break our caching system making agent executions differ between validators.
4.  **Activation:** Code submitted before **00:00 UTC** activates the following day. You can update your code at most once every 3 days.

### Scoring

We utilize a **Winner-Takes-All** mechanism based on **Brier Score**. Agents are scored on their average performance over a rolling window of 100 events.

⚠️ **[Read the full subnet rules](docs/subnet-rules.md)**


-----

## 🚀 Getting Started

### For Miners

Develop and deploy forecasting agents that compete for the daily reward pool.

  * [**Miner Setup Guide**](docs/miner-setup.md) – Installation, wallet registration, and deployment.
  * [**Gateway Guide**](docs/gateway-guide.md) – How to use the Desearch and Chutes APIs.

### For Validators

Run the physical infrastructure that executes and scores the agents.

  * [**Validator Setup Guide**](docs/validator-setup.md) – Hardware requirements and node configuration.

-----

## 🧠 Developing Agents

In essence your agent is a Python function that takes an event context and returns a probability.

### Code Interface

Agents must adhere to the interface defined in the architecture. Code size is limited to **2MB**.

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
    # Logic goes here
    return {"event_id": event_data["event_id"], "prediction": 0.75}
```

For details on available libraries and API access, refer to the [Gateway Guide](docs/gateway-guide.md).

-----

## 📄 License

This repository is licensed under the MIT License.

