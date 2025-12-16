# Miner Setup Guide

> **⚠️ IMPORTANT ARCHITECTURE CHANGE**
> Miners NO LONGER run nodes or servers. You submit Python agent CODE via API.
> NO infrastructure costs. NO API costs. Just write your forecasting logic.

## Overview

This guide walks you through:
1. Setting up your development environment
2. Creating and registering a Bittensor wallet
3. Writing your forecasting agent code
4. Testing your agent locally
5. Submitting your agent to the network

For competition rules and constraints, see [subnet-rules.md](./subnet-rules.md).
For system architecture details, see [architecture.md](./architecture.md).
For gateway API reference (Chutes AI, Desearch AI), see [gateway-guide.md](./gateway-guide.md).

The key rules to follow as a miner are the following:
- **The sandbox times out after 150s**
- **The total cost limit on API calls is $0.02**
- **DO NOT include dynamic timestamps or random data in prompts to make sure our caching system is hit across different validator executions**.
- **A forecasting agent can only be updated at most once every 3 days**

All events are currently 3 days events. The length of the immunity period is about 4 days to ensure any time before registration. 


---

# System Requirements

**For Local Development & Testing:**
- Python 3.11+
- Text editor or IDE
- `numi` CLI tool (installed via this repo)
- **Chutes AI API key** (for local testing with LLMs)
- **Desearch AI API key** (for local testing with web/Twitter search)

**Get API Keys:**
- Chutes AI: https://chutes.ai/app
- Desearch AI: https://desearch.ai/

**Note:** API keys are only needed for **local testing**. In production, validators provide API access via gateway at no cost to you.

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
2. Get testnet TAO from https://app.minersunion.ai/testnet-faucet
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

    Args:
        event_data: {
            "event_id": str,        # Unique event identifier
            "title": str,           # Short event title
            "description": str,     # Full event description
            "cutoff": str,          # ISO 8601 datetime (prediction deadline)
            "metadata": dict        # Event-specific data
        }

    Returns:
        {
            "event_id": str,        # Echo back from input
            "prediction": float     # Probability in [0.0, 1.0]
        }
    """
    prediction = 0.5  # Your logic here

    return {
        "event_id": event_data["event_id"],
        "prediction": prediction
    }
```

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

### LLM-Based Agent (Using Chutes AI)

**Important:** All agents MUST use the proxy URL and include `RUN_ID` in their requests.

```python
import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI

# Required: Get proxy URL and run ID from environment
PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
RUN_ID = os.getenv("RUN_ID")  # Required - validator provides this

# Validate required environment variables
if not RUN_ID:
    raise ValueError("RUN_ID environment variable is required but not set")

# Initialize LLM pointing to gateway
CHUTES_URL = f"{PROXY_URL}/api/gateway/chutes"

llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3-0324",
    base_url=CHUTES_URL,
    api_key="not-needed",
    extra_body={"run_id": RUN_ID},
)

def agent_main(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """LLM-based forecasting agent."""

    prompt = f"""You are a forecasting expert. Analyze this event and provide a probability between 0 and 1.

    Event: {event_data['description']}
    Cutoff: {event_data['cutoff']}

    Return ONLY a number between 0 and 1."""

    response = llm.invoke(prompt)
    prediction_text = response.content.strip()
    prediction = float(prediction_text)

    # Ensure valid range
    prediction = max(0.0, min(1.0, prediction))

    return {
        "event_id": event_data["event_id"],
        "prediction": prediction
    }
```

### Using Desearch (Web/Twitter Search)

```python
import os
import httpx
from typing import Dict, Any

# Required: Get proxy URL and run ID
PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
RUN_ID = os.getenv("RUN_ID")

if not RUN_ID:
    raise ValueError("RUN_ID environment variable is required but not set")

DESEARCH_URL = f"{PROXY_URL}/api/gateway/desearch"

def agent_main(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """Uses Desearch to gather information."""

    # Search for relevant information
    payload = {
        "prompt": f"Search for information about: {event_data['title']}",
        "tools": ["WEB"],  # or ["TWITTER"]
        "model": "NOVA",
        "streaming": False,
        "count": 10,
        "run_id": str(RUN_ID),
    }

    response = httpx.post(
        f"{DESEARCH_URL}/ai/search",
        json=payload,
        timeout=60.0,
    )

    results = response.json()

    # Analyze results and compute prediction
    prediction = analyze_results(results, event_data)

    return {
        "event_id": event_data["event_id"],
        "prediction": prediction
    }

def analyze_results(results, event_data):
    # Your analysis logic here
    return 0.5
```

## Important Notes

1. **Always use `SANDBOX_PROXY_URL`** - Never hardcode API URLs
2. **Always include `RUN_ID`** - Required for tracking and authentication
3. **Check hot models** - Visit https://chutes.ai/app to see available models before using them
4. **Implement retry logic** - Handle API errors with proper fallback strategies

## Best Practices

### Error Handling

Always implement robust error handling for API calls. Chutes AI can return these errors:

- **503 Service Unavailable** - Cold model (no active instances), implement exponential backoff
- **404 Not Found** - Model doesn't exist, check https://chutes.ai/app for available models
- **429 Too Many Requests** - Rate limit exceeded, implement exponential backoff

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

**Leave buffer time for retries** - With a 150-second timeout, plan your execution:

- Multiple retries: Account for exponential backoff delays
- Fallback logic: Always have a quick fallback (return 0.5) if time runs out

**Example timing strategy:**

```python
import time

start_time = time.time()
timeout = 140  # Leave 10s buffer before hard 150s limit

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
```

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

# Submitting Your Agent

Place your agent in the expected directory:

```bash
mkdir -p neurons/miner/agents
cp my_agent.py neurons/miner/agents/
```

Submit using the CLI:

```bash
# Interactive mode (recommended)
numi upload-agent

# Or specify all options
numi upload-agent \
  --agent-file my_agent.py \
  --env test \
  --wallet miner \
  --hotkey default \
  --name "My Forecaster v1"
```

The CLI will guide you through the process - just follow the prompts!

**Upload confirmation:**

```
✓ Upload successful!
Agent ID: [generated_id]
Network: TEST

⚠️  Remember to link services for this new code!
Run: numi services link
```

## Linking Services

After uploading your agent, link your Desearch account to cover Desearch API costs:

```bash
# Link Desearch (required for production use)
numi services link desearch
```

You'll be prompted for:
- Your Desearch API key (get from https://console.desearch.ai)
- Coldkey password (to sign the linking)

**Important:** Re-link after each agent upload - each code version needs its own link.

Check your linked services anytime:
```bash
numi services list
```

**Note:** Currently, only Desearch costs are billed to your account. In the future, all API costs will transition to miner accounts, allowing us to add more services and increase cost limits for better forecasting capabilities.

## Activation Schedule

⚠️ **Important:** Submitted code activates at **next 00:00 UTC**.

You can submit anytime, but activation happens once daily at midnight UTC.

## Complete CLI Command Reference

```bash
# Agent Management
numi upload-agent          # Submit agent to network
numi list-agents           # List your uploaded agents
numi inspect-agent         # View/download agent code

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
2. Review [architecture.md](./architecture.md) for system details
3. Check example agents in `neurons/miner/agents/`
