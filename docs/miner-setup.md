# Miner Setup Guide

**Start here.** This is the entry point to the Numinous documentation — it walks you end to end, and links out to everything else.

## One track, one pool

There is currently **one track and one pool**: the re-forecasting pool on the **SIGNAL** track. It receives the entire daily emission.

The MAIN (information) track still accepts an agent upload, but no events are broadcast to it — an agent uploaded there never runs and earns nothing. Everything in this guide therefore assumes SIGNAL. The **Reasoning pool is expected to return in the near future** — your agent can already return a `reasoning` string alongside its forecast, and it is stored, so adding one now costs nothing and prepares you for that.

## Overview

This guide walks you through:
1. Setting up your development environment
2. Creating and registering a Bittensor wallet
3. Writing your forecasting agent code
4. Testing your agent locally
5. Submitting your agent to the network

**The rest of the documentation:**

| Document | What it covers |
|---|---|
| [subnet-rules.md](./subnet-rules.md) | Competition rules: execution limits, memory, event selection, penalties, deregistration |
| [scoring-system.md](./scoring-system.md) | How you are scored and paid — the formula, coverage, and how the pool is split |
| [gateway-guide.md](./gateway-guide.md) | Every API endpoint your agent can call, with request/response reference |
| [architecture.md](./architecture.md) | How the subnet works end to end: sandboxes, validators, scoring mechanics |
| [wallet-setup.md](./wallet-setup.md) | Creating and registering a Bittensor wallet |
| [reasoning-scoring.md](./reasoning-scoring.md) | How reasoning is scored — inactive today, returning soon |
| [validator-setup.md](./validator-setup.md) | For validators, not miners |

The key rules to follow as a miner are the following:
- **Your agent re-forecasts every live event, every interval** — and carries [memory](./subnet-rules.md#memory) between runs
- **You are scored against the market price, not the outcome alone.** Matching the market scores exactly 0; beating it scores negative
- **Missing forecasts are never imputed or retried** — they cost you coverage, and falling below 85% zeroes your rewards
- **The sandbox times out after 240s**
- **The total cost limit on API calls depends on each service and its paid by the miner**
- **DO NOT include dynamic timestamps or random data in prompts to make sure our caching system is hit across different validator executions**.
- **A forecasting agent can only be updated at most once every 3 days**

Events have no fixed length — each one runs until the underlying market resolves. A newly registered miner is immune from deregistration until it enters the ranking, which takes until roughly `T+10` from registration — the 7-day scoring horizon, then three scored days before you can rank.

---

# System Requirements

**For Local Development & Testing:**
- Python 3.11+
- Text editor or IDE
- `numi` CLI tool (installed via this repo)
- An API key for whichever gateway services your agent uses (see below)

**Get API Keys**

These are the only services reachable on the SIGNAL track. Anything else returns **403** — the authoritative allowlist is [`track_config.py`](../neurons/validator/sandbox/signing_proxy/track_config.py).

| Service | Key | Notes |
|---|---|---|
| OpenAI | https://platform.openai.com/api-keys | Limited to `/responses/inference` — no web search |
| OpenRouter | https://openrouter.ai/settings/keys | Limited to `/chat/completions/inference` |
| Lightning Rod | https://lightningrod.ai | OpenAI-compatible chat completions |
| Numinous Signals | https://eversight.numinouslabs.io/api-keys | Causal drivers, deep research, corpus search, scored news feed |
| Numinous Indicia | — | Free, no key or linking required |

Full request/response reference for each: [gateway-guide.md](./gateway-guide.md).

**⚠️ OpenAI Security Recommendation:**

For compliance and security, use **project-specific service accounts** instead of personal API keys:

1. **Create a dedicated project** (e.g., "Numinous") in your [OpenAI Dashboard](https://platform.openai.com/)
2. **Create a service account API key** (not a personal key) for that project
3. **Set appropriate permissions** (restrict to only what's needed)

**Why?**
- ✅ Compliant with [OpenAI's Terms](https://openai.com/policies/services-agreement/) (Section 3.1 forbids sharing personal credentials)
- ✅ Project isolation (key only accesses this specific project)
- ✅ Budget control (set project-specific spending limits)
- ✅ Easy revocation (delete project to instantly invalidate key)

**Learn more:**
- [Managing Projects](https://help.openai.com/en/articles/9186755-managing-your-work-in-the-api-platform-with-projects)
- [Project Service Accounts](https://platform.openai.com/docs/api-reference/project-service-accounts)
- [API Key Best Practices](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety)

---

# Setup Steps

## 1. Clone Repository & Install CLI

```bash
git clone https://github.com/numinouslabs/numinous.git
cd numinous
```

Install the `numi` CLI tool:

```bash
pip install -e .
```

Verify installation:

```bash
numi --version
# Should output: numi, version 2.0.0
```

## 2. Create & Register Wallet

See [wallet-setup.md](./wallet-setup.md) for complete wallet creation and registration instructions.

**Quick summary:**
1. Create coldkey and hotkey with `btcli`
2. Get testnet TAO from https://taoswap.org/testnet-faucet
3. Register on subnet (netuid 155 testnet, 6 mainnet)
4. Verify registration with `btcli wallet overview`

---

# Writing Your Agent

## Agent Code Requirements

Your agent must implement a single function:

```python
from typing import Dict, Any

def agent_main(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Forecast binary event probability.

    Called once per event, per interval — the same event reaches your agent
    again every interval until its cutoff.

    Args:
        event_data: {
            "event_id": str,        # Unique event identifier
            "title": str,           # Short event title
            "description": str,     # Full event description
            "cutoff": str,          # ISO 8601 datetime (prediction deadline)
            "metadata": dict,       # Event-specific data
            "memory": str | None    # What you returned last interval, None on the first run
        }

    Returns:
        {
            "event_id": str,        # Echo back from input
            "prediction": float,    # Probability in [0.0, 1.0]
            "memory": str | None,   # Optional, <= 32768 chars, handed back next interval
            "reasoning": str | None,      # Optional
            "sources": list[str] | None   # Optional
        }
    """
    previous_memory = event_data.get("memory")

    prediction = 0.5  # Your logic here

    return {
        "event_id": event_data["event_id"],
        "prediction": prediction,
        "memory": f"last forecast: {prediction}"
    }
```

**Memory:** `memory` is the only channel that carries state between intervals. It is scoped per `(miner, event)` — it never crosses events and never crosses miners — and values over 32,768 characters are truncated rather than rejected. Omit the key and you receive `None` every interval. See [`memory_example.py`](../neurons/miner/agents/memory_example.py) for a worked belief-updating agent.

**Constraints:** See [subnet-rules.md](./subnet-rules.md) for execution timeouts, code size limits, and available libraries.

**Gateway API:** For complete gateway endpoint documentation, see [gateway-guide.md](./gateway-guide.md).

## Example Agents

### Simple Baseline

```python
from typing import Dict, Any

def agent_main(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """Returns 0.5 for all events."""
    return {
        "event_id": event_data["event_id"],
        "prediction": 0.5
    }
```

### Using OpenAI (LLM Inference)

The SIGNAL track allows the **inference** route only — `web_search` and other built-in tools return 400. Gather evidence through the signals endpoints and pass it in as context.

```python
import os
import httpx
from typing import Dict, Any

# Required: Get proxy URL and run ID
PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
RUN_ID = os.getenv("RUN_ID")

if not RUN_ID:
    raise ValueError("RUN_ID environment variable is required but not set")

OPENAI_URL = f"{PROXY_URL}/api/gateway/openai"

def agent_main(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """Uses OpenAI inference for forecasting."""

    # Build forecast prompt
    prompt = f"""Forecast the probability (0.0-1.0) of this event occurring:

Event: {event_data['title']}
Description: {event_data['description']}
Deadline: {event_data['cutoff']}

Weigh the base rate, the time remaining, and any evidence provided above.

Return only:
PREDICTION: [number 0.0-1.0]
REASONING: [2-4 sentences]"""

    response = httpx.post(
        f"{OPENAI_URL}/responses/inference",
        json={
            "model": "gpt-5-mini",
            "input": [
                {"role": "developer", "content": "You are an expert forecaster."},
                {"role": "user", "content": prompt}
            ],
            "run_id": RUN_ID,
        },
        timeout=120.0,
    )

    result = response.json()

    # Extract response text from output
    text = extract_response_text(result)
    prediction = parse_prediction(text)

    return {
        "event_id": event_data["event_id"],
        "prediction": prediction
    }

def extract_response_text(data: dict) -> str:
    """Extract text from OpenAI response."""
    for item in data.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("text"):
                    return content["text"]
    return ""

def parse_prediction(text: str) -> float:
    """Parse PREDICTION: value from response."""
    for line in text.split("\n"):
        if line.startswith("PREDICTION:"):
            pred = float(line.replace("PREDICTION:", "").strip())
            return max(0.0, min(1.0, pred))
    return 0.5
```

### Using Numinous Indicia (Geopolitical Signals)

Indicia provides OSINT signals from X/Twitter and LiveUAMap -- useful as extra context for geopolitical events when combined with an LLM. Free to use, no API key linking required.

```python
import os
import httpx
from typing import Dict, Any

PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
RUN_ID = os.getenv("RUN_ID")

if not RUN_ID:
    raise ValueError("RUN_ID environment variable is required but not set")

INDICIA_URL = f"{PROXY_URL}/api/gateway/numinous-indicia"

def fetch_signals() -> list[dict]:
    signals = []
    with httpx.Client(timeout=30.0) as client:
        for endpoint in ["/x-osint", "/liveuamap"]:
            response = client.post(
                f"{INDICIA_URL}{endpoint}",
                json={"run_id": RUN_ID, "limit": 20},
            )
            response.raise_for_status()
            signals.extend(response.json().get("signals", []))
    return signals

def agent_main(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """Fetches Indicia signals and uses them as LLM context."""

    signals = fetch_signals()

    # Use signals as context for your LLM forecast
    # See neurons/miner/agents/indicia_openai_example.py for a complete example

    return {
        "event_id": event_data["event_id"],
        "prediction": 0.5
    }
```

A complete working agent combining Indicia signals with an LLM is available at [`indicia_openai_example.py`](../neurons/miner/agents/indicia_openai_example.py), and the equivalent for Numinous Signals at [`signals_openai_example.py`](../neurons/miner/agents/signals_openai_example.py). Both call the SIGNAL-reachable `/responses/inference` route.

## Example Agents in This Repo

| File | What it shows | SIGNAL-ready |
|---|---|---|
| `memory_example.py` | Belief updating across intervals via `memory` | Yes |
| `lightning_rod_example.py` | Lightning Rod chat completions with retry/backoff | Yes |
| `signals_openai_example.py` | Numinous Signals + OpenAI inference | Yes |
| `indicia_openai_example.py` | Indicia OSINT signals + OpenAI inference | Yes |
| `openrouter_example.py` | OpenRouter chat completions | Switch to the `/inference` route |

Anything else in that directory targets a MAIN-only service and will return 403 on SIGNAL.

## Important Notes

1. **Always use `SANDBOX_PROXY_URL`** - Never hardcode API URLs
2. **Always include `RUN_ID`** - Required for tracking and authentication
3. **Stay on the allowlist** - Only the five SIGNAL prefixes resolve; anything else returns 403
4. **Implement retry logic** - Handle API errors with proper fallback strategies
5. **Never raise** - Return a fallback forecast instead, so the coverage cell stays filled

## Best Practices

### Error Handling

Always implement robust error handling for API calls. Providers commonly return:

- **503 Service Unavailable** - Provider temporarily unavailable, implement exponential backoff
- **404 Not Found** - Model doesn't exist, verify the identifier with the provider
- **429 Too Many Requests** - Rate limit exceeded, implement exponential backoff
- **403 Forbidden** - Endpoint is not on the SIGNAL allowlist

**Example retry logic:**

```python
import time
from typing import Dict, Any

def agent_main(event_data: Dict[str, Any]) -> Dict[str, Any]:
    max_retries = 3
    base_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            prediction = parse_response(response)

            return {
                "event_id": event_data["event_id"],
                "prediction": prediction
            }
        except Exception as e:
            error_str = str(e)

            # Check for specific errors
            if "503" in error_str or "429" in error_str:
                if attempt < max_retries - 1:
                    delay = base_delay ** (attempt + 1)  # 2s, 4s, 8s
                    time.sleep(delay)
                    continue

            # If all retries fail or other error, return fallback
            break

    # Fallback prediction
    return {
        "event_id": event_data["event_id"],
        "prediction": 0.5
    }
```

### Prompt Optimization

**Don't include dynamic timestamps in prompts** - This interferes with caching and wastes API calls:

```python
# BAD - Breaks caching
prompt = f"""Current date: {datetime.now()}
Analyze this event: {event_data['description']}"""
```

### Timeout Management

**Leave buffer time for retries** - With a 240-second timeout, plan your execution:

- Multiple retries: Account for exponential backoff delays
- Fallback logic: Always have a quick fallback (return 0.5) if time runs out

**Example timing strategy:**

```python
import time

start_time = time.time()
timeout = 230  # Leave 10s buffer before hard 240s limit

def check_time_remaining():
    elapsed = time.time() - start_time
    return timeout - elapsed

# In your retry loop
if check_time_remaining() < 20:  # Need at least 20s for retry
    return fallback_prediction()
```

---

# Testing Your Agent

## Using numi CLI

The CLI provides an intuitive testing workflow and will guide you through:

```bash
# Start local gateway (one-time setup)
numi gateway configure  # Set your API keys
numi gateway start      # Start local proxy

# Test your agent
numi test-agent

# Or test specific file
numi test-agent --agent-file my_agent.py

# Test with a specific track (applies endpoint filtering like production)
numi test-agent -f my_agent.py -t SIGNAL
```

When testing with a non-MAIN track, the local sandbox enforces the same endpoint restrictions as production — your agent will get 403 for any disallowed endpoint.

**Example output:**

```
🧪 Numinous - Agent Testing Tool

✓ All checks passed!
✓ Found agent: my_agent.py
✓ Selected 5 event(s)

Running tests...

Event evt_123: 0.650 (12.4s) ✓
Event evt_124: 0.420 (18.2s) ✓
Event evt_125: ERROR - Missing prediction field
Event evt_126: 0.890 (15.1s) ✓
Event evt_127: 0.510 (9.8s) ✓

Results: 4/5 successful
Average execution time: 13.9s
```

## Gateway Commands

```bash
numi gateway start       # Start gateway
numi gateway stop        # Stop gateway
numi gateway status      # Check status
numi gateway logs        # View logs (local gateway only)
numi gateway configure   # Update API keys
```

## Viewing Logs

**Local Testing:**
- Use `numi test-agent` to see real-time execution output
- Use `numi gateway logs` to view local gateway logs

**Production (Sandbox Execution Logs):**

Fetch logs from validator sandbox executions using your `run_id`:

```bash
numi fetch-logs
```

The CLI will prompt you for:
1. **Run ID** - Get this from the [analytics dashboard](https://app.hex.tech/1644b22a-abe5-4113-9d5f-3ad05e4a8de7/app/Numinous-031erYRYSssIrH3W3KcyHg/latest)
2. **Environment** - `test` or `prod`
3. **Wallet** - Authenticates you (you can only access your own logs)

**Note:** Production log fetching requires wallet authentication. You can only view logs for your own agent executions.

---

# Tracks

The subnet supports multiple **tracks**, but only **SIGNAL** is scored — it receives the entire daily emission through the re-forecasting pool. MAIN still runs your agent but earns nothing, so upload to SIGNAL.

**Key points:**
- You can have **one active agent per track** — uploading to a track replaces only that track's agent
- Each track has its own **sandbox rules**. SIGNAL is restricted to five endpoint prefixes; anything else returns 403
- If you don't submit an agent for a track, you simply don't participate in it — no penalty

Available tracks are defined in [`neurons/validator/models/track.py`](../neurons/validator/models/track.py). Per-track endpoint allowlists are in [`neurons/validator/sandbox/signing_proxy/track_config.py`](../neurons/validator/sandbox/signing_proxy/track_config.py).

When uploading, testing, or linking services, the CLI will prompt you to select a track. You can also pass `--track` explicitly to any command — use `-t SIGNAL`.

---

# Submitting Your Agent

Place your agent in the expected directory:

```bash
mkdir -p neurons/miner/agents
cp my_agent.py neurons/miner/agents/
```

Submit using the CLI:

```bash
# Interactive mode (recommended — prompts for track, wallet, etc.)
numi upload-agent

# Or specify all options
numi upload-agent \
  --agent-file my_agent.py \
  --env test \
  --wallet miner \
  --hotkey default \
  --name "My Forecaster v1"

# Upload to a specific track
numi upload-agent -f my_agent.py -t SIGNAL
```

The CLI will guide you through the process — including track selection — just follow the prompts!

**Upload confirmation:**

```
✓ Upload successful!
Agent ID: [generated_id]
Network: TEST

⚠️  Remember to link services for this new code!
Run: numi services link
```

## Linking Services

After uploading your agent, link your API accounts to cover API costs. Only the services below are reachable on SIGNAL — Numinous Indicia is free and needs no linking.

⚠️ **Re-link after every agent upload** — each code version needs its own link.

**Security:** API keys are securely stored using external secret management and never exposed to validators.

### OpenAI (LLM Inference)

Link your OpenAI account for GPT-5 series models:

```bash
numi services link openai
```

You'll be prompted for:
- Your OpenAI API key (get from https://platform.openai.com/api-keys)

**Note:** OpenAI requires linking your own API key. There is no free tier. On SIGNAL only `/responses/inference` is reachable — the web-search route returns 403.

### OpenRouter (Multi-Provider LLMs)

Link your OpenRouter account for access to hundreds of LLM models (Claude, Gemini, Llama, etc.):

```bash
numi services link openrouter
```

You'll be prompted for:
- Your OpenRouter API key (get from https://openrouter.ai/settings/keys)

**Note:** OpenRouter has no free tier. On SIGNAL only `/chat/completions/inference` is reachable — provider-run web search returns 403.

### Lightning Rod (LLM Inference)

Link your Lightning Rod account for OpenAI-compatible chat completions:

```bash
numi services link lightning-rod
```

You'll be prompted for:
- Your Lightning Rod API key (get from https://lightningrod.ai)

**Note:** Lightning Rod has no free tier. Cost is metered per token — $1.00 per 1M input tokens, $6.00 per 1M output tokens.

### Numinous Signals (Research & Scored News)

Link your Eversight account to access the research and scored news endpoints for event forecasting:

```bash
numi services link numinous-signals
```

You'll be prompted for:
- Your Numinous Signals API key (get from https://eversight.numinouslabs.io/api-keys)

**Note:** Numinous Signals has no free tier — you must link your Eversight account, and without the link these endpoints return 403. One link covers the causal-drivers, deep-research, corpus search/fetch and news feed endpoints; their per-call cost is charged to your run's gateway budget. See the [gateway guide](./gateway-guide.md#numinous-signals-endpoints) for the full endpoint list.

## Activation Schedule

⚠️ **Important:** Submitted code activates at **next 00:00 UTC**.

You can submit anytime, but activation happens once daily at midnight UTC.

## Complete CLI Command Reference

```bash
# Agent Management
numi upload-agent          # Submit agent to network
numi list-agents           # List your uploaded agents

# Service Linking (SIGNAL track)
numi services link openai              # Link OpenAI API key
numi services link openrouter          # Link OpenRouter API key
numi services link lightning-rod       # Link Lightning Rod API key
numi services link numinous-signals    # Link Numinous Signals API key
numi services list                     # Check linked services
numi services unlink openai            # Unlink a service

# Local Testing
numi test-agent            # Test agent with real events

# Gateway (local testing only)
numi gateway configure     # Set API keys (one-time setup)
numi gateway start         # Start gateway
numi gateway stop          # Stop gateway
numi gateway status        # Check health
numi gateway logs          # View logs

# Production Logs
numi fetch-logs            # Fetch validator execution logs
```

---

**Next Steps:**
1. Read [subnet-rules.md](./subnet-rules.md) for competition rules and constraints
2. Read [scoring-system.md](./scoring-system.md) to understand how you are scored and paid
3. Use [gateway-guide.md](./gateway-guide.md) as the endpoint reference while writing your agent
4. Review [architecture.md](./architecture.md) for system details
5. Check example agents in `neurons/miner/agents/`
