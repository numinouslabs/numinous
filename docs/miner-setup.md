# Miner Setup Guide

## Overview

This guide walks you through:
1. Setting up your development environment
2. Creating and registering a Bittensor wallet
3. Writing your forecasting agent code
4. Testing your agent locally
5. Submitting your agent to the network

For competition rules and constraints, see [subnet-rules.md](./subnet-rules.md).
For system architecture details, see [architecture.md](./architecture.md).
For gateway API reference (Chutes AI, Desearch AI, Numinous Indicia, etc.), see [gateway-guide.md](./gateway-guide.md).
For how miner-submitted reasoning is scored (and why a large share of emissions depends on it), see [reasoning-scoring.md](./reasoning-scoring.md).

The key rules to follow as a miner are the following:
- **The sandbox times out after 240s**
- **The total cost limit on API calls depends on each service and its paid by the miner**
- **DO NOT include dynamic timestamps or random data in prompts to make sure our caching system is hit across different validator executions**.
- **A forecasting agent can only be updated at most once every 3 days**

All events are currently 3 days events. The length of the immunity period is 7 days to ensure any time before registration.

---

# System Requirements

**For Local Development & Testing:**
- Python 3.11+
- Text editor or IDE
- `numi` CLI tool (installed via this repo)
- **Chutes AI API key** (for local testing with LLMs)
- **Desearch AI API key** (for local testing with web/Twitter search)
- **OpenAI API key** (for local testing with GPT-5 models)
- **Perplexity API key** (for local testing with reasoning LLMs)
- **Vericore API key** (for local testing with statement verification)
- **OpenRouter API key** (for local testing with multi-provider LLM access)
- **Numinous Signals API key** (for local testing with scored news signals)
- **Unusual Whales API key** (for local testing with financial news headlines)

**Get API Keys:**
- Chutes AI: https://chutes.ai/app
- Desearch AI: https://desearch.ai/
- OpenAI: https://platform.openai.com/api-keys
- Perplexity: https://www.perplexity.ai/settings/api
- Vericore: https://vericore.ai
- LunarCrush: https://lunarcrush.com
- OpenRouter: https://openrouter.ai/settings/keys
- Numinous Signals: https://eversight.numinouslabs.io/api-keys
- Unusual Whales: https://unusualwhales.com/pricing?product=api

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

### Using OpenAI (LLM with Web Search)

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
    """Uses OpenAI with built-in web search for forecasting."""

    # Build forecast prompt
    prompt = f"""Forecast the probability (0.0-1.0) of this event occurring:

Event: {event_data['title']}
Description: {event_data['description']}
Deadline: {event_data['cutoff']}

Before making your forecast, systematically research:
1. Search for recent news and developments
2. Search for expert analysis and predictions
3. Search for historical data or precedents

Return only:
PREDICTION: [number 0.0-1.0]
REASONING: [2-4 sentences]"""

    # Call OpenAI with web_search tool
    response = httpx.post(
        f"{OPENAI_URL}/responses",
        json={
            "model": "gpt-5-mini",
            "input": [
                {"role": "developer", "content": "You are an expert forecaster."},
                {"role": "user", "content": prompt}
            ],
            "tools": [{"type": "web_search"}],  # Enable web search
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

### Using Perplexity

```python
import os
import httpx
from typing import Dict, Any

PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
RUN_ID = os.getenv("RUN_ID")

if not RUN_ID:
    raise ValueError("RUN_ID environment variable is required but not set")

PERPLEXITY_URL = f"{PROXY_URL}/api/gateway/perplexity"

def agent_main(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """Uses Perplexity reasoning LLM with web search for forecasting."""

    prompt = f"""Forecast the probability (0.0-1.0) of this event occurring:

Event: {event_data['title']}
Description: {event_data['description']}
Deadline: {event_data['cutoff']}

Search for recent information and provide:
PREDICTION: [number 0.0-1.0]
REASONING: [2-4 sentences]"""

    response = httpx.post(
        f"{PERPLEXITY_URL}/chat/completions",
        json={
            "model": "sonar-reasoning-pro",
            "messages": [
                {"role": "system", "content": "You are an expert forecaster."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "search_recency_filter": "week",
            "run_id": RUN_ID,
        },
        timeout=120.0,
    )

    result = response.json()

    text = result["choices"][0]["message"]["content"]
    citations = result.get("citations", [])

    prediction = parse_prediction(text)

    return {
        "event_id": event_data["event_id"],
        "prediction": prediction
    }

def parse_prediction(text: str) -> float:
    """Parse PREDICTION: value from response."""
    for line in text.split("\n"):
        if line.startswith("PREDICTION:"):
            pred = float(line.replace("PREDICTION:", "").strip())
            return max(0.0, min(1.0, pred))
    return 0.5
```

### Using Vericore (Statement Verification)

```python
import os
import httpx
from typing import Dict, Any

PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
RUN_ID = os.getenv("RUN_ID")

if not RUN_ID:
    raise ValueError("RUN_ID environment variable is required but not set")

VERICORE_URL = f"{PROXY_URL}/api/gateway/vericore"

def agent_main(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """Uses Vericore to verify event statement against web evidence."""

    statement = f"{event_data['title']}. {event_data['description']}"

    response = httpx.post(
        f"{VERICORE_URL}/calculate-rating",
        json={
            "statement": statement,
            "run_id": RUN_ID,
        },
        timeout=120.0,
    )

    result = response.json()
    summary = result["evidence_summary"]

    # Use evidence metrics to derive prediction
    entailment = summary.get("entailment", 0.0)
    contradiction = summary.get("contradiction", 0.0)
    neutral = summary.get("neutral", 0.0)

    total = entailment + contradiction + neutral
    if total > 0:
        prediction = entailment / total
    else:
        prediction = 0.5

    prediction = max(0.0, min(1.0, prediction))

    return {
        "event_id": event_data["event_id"],
        "prediction": prediction
    }
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

A complete working agent combining Indicia signals with OpenAI web search is available at `neurons/miner/agents/indicia_openai_example.py`.

### Using Indicia RSS Feeds

Indicia also exposes a curated RSS aggregator at `https://indicia.numinouslabs.io/rss/articles`, reachable through the [public data proxy](./gateway-guide.md#public-data-proxy). The domain is whitelisted as `indicia_rss` (free, no API key linking). Use it to pull news articles grouped by pipeline, category, or feed variant.

**Quick reference**

| Need | Query params |
|---|---|
| All geopolitical news | `pipeline=geopolitical` |
| One category only | `categories=crypto` |
| Multi-category | `categories=middleeast,energy,intel` |
| Specific variant (e.g. all intel) | `variants=intel` |
| Pipeline with cap | `pipeline=middleeast&max_articles=50` |

**Pipeline groups**

| Pipeline | Categories included |
|---|---|
| `geopolitical` | politics, us, europe, middleeast, asia, africa, latam, gov, thinktanks, crisis, intel |
| `middleeast` | middleeast, intel, crisis |
| `commodity` | commodity_news, gold_silver, energy, mining_news, critical_minerals, base_metals, mining_companies, supply_chain, commodity_regulation |
| `finance` | finance, markets, forex, bonds, commodities, crypto, centralbanks, economic, derivatives, fintech, regulation, institutional, analysis, gcc |
| `tech` | tech, ai, startups, vcblogs, regional_startups, security, policy, cloud, dev, hardware |

**Variants:** `full` (general news), `tech`, `finance`, `commodity`, `intel`.

**Example**

```python
import json
import os
import httpx

PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
RUN_ID = os.getenv("RUN_ID")

INDICIA_RSS_URL = "https://indicia.numinouslabs.io/rss/articles"

def fetch_indicia_rss(**query_params) -> list[dict]:
    response = httpx.post(
        f"{PROXY_URL}/api/gateway/public-data/fetch",
        json={
            "run_id": RUN_ID,
            "url": INDICIA_RSS_URL,
            "method": "GET",
            "query_params": {k: str(v) for k, v in query_params.items()},
            "timeout": 30.0,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    return json.loads(response.json()["response_body"])

# By pipeline group
geopolitical = fetch_indicia_rss(pipeline="geopolitical")

# By category (single or comma-separated)
crypto_news = fetch_indicia_rss(categories="crypto,fintech")

# By variant
intel_only = fetch_indicia_rss(variants="intel")

# Pipeline with article cap
middle_east = fetch_indicia_rss(pipeline="middleeast", max_articles=50)
```

Notes:
- The proxy enforces a `https://indicia.numinouslabs.io/rss/articles` URL prefix — only this path is reachable, not other Indicia endpoints.
- Responses are capped at 5MB; use `pipeline` or `max_articles` to keep payloads reasonable when pulling broad slices like `ALL_FEEDS` (~230 feeds).
- Cost is `$0.00`. Run `numi services sources` to confirm the source is currently whitelisted.

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

A miner can participate in multiple **tracks** simultaneously. Each track is an independent competition — you upload a separate agent per track, and each one is scored and weighted independently against other agents on that same track. Think of it as running multiple miners from a single registration.


**Key points:**
- You can have **one active agent per track** — uploading to a track replaces only that track's agent
- Each track has its own **scoring pool** — you compete only against other miners on the same track
- Tracks may have different **sandbox rules** (e.g. which gateway endpoints are accessible)
- Service credentials linked for the MAIN track **fall back** to all other tracks — you only need track-specific credentials if you want separate API keys
- If you don't submit an agent for a track, you simply don't participate in it — no penalty

Available tracks are defined in [`neurons/validator/models/track.py`](../neurons/validator/models/track.py). Per-track sandbox rules (endpoint allowlists) are in [`neurons/validator/sandbox/signing_proxy/track_config.py`](../neurons/validator/sandbox/signing_proxy/track_config.py).

When uploading, testing, or linking services, the CLI will prompt you to select a track. You can also pass `--track` explicitly to any command.

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

After uploading your agent, link your API accounts to cover API costs for LLM inference and search.

**Tracks & credentials:** Credentials linked for the MAIN track are used as a fallback for all other tracks. You only need to link track-specific credentials if you want separate API keys per track. Use `--track` to link for a specific track, or the CLI will prompt you.

**Security:** API keys are securely stored using external secret management and never exposed to validators.

### Chutes AI (LLM Inference)

Link your Chutes account to access higher budget for LLM API calls:

```bash
numi services link chutes
```

You'll be prompted for:
- Your Chutes API key (get from https://chutes.ai/app)

**Cost Tiers:**
- Free tier (default): $0.01 per agent run
- Paid tier (your key): $0.10 per agent run

### Desearch AI (Search & Data)

Link your Desearch account to cover search API costs:

```bash
numi services link desearch
```

You'll be prompted for:
- Your Desearch API key (get from https://console.desearch.ai)
- Coldkey password (to sign the linking)

**Cost Tiers:**
- Free tier (default): $0.01 per agent run
- Paid tier (your key): $0.10 per agent run

### OpenAI (LLM Inference)

Link your OpenAI account for GPT-5 series models with web search:

```bash
numi services link openai
```

You'll be prompted for:
- Your OpenAI API key (get from https://platform.openai.com/api-keys)

**Note:** OpenAI requires linking your own API key. There is no free tier - you must link your account to use OpenAI models.

### Perplexity

Link your Perplexity account for reasoning LLMs with web search:

```bash
numi services link perplexity
```

You'll be prompted for:
- Your Perplexity API key (get from https://www.perplexity.ai/settings/api)

**Note:** Perplexity has no free tier. You must link your account to use Perplexity models.

### Vericore (Statement Verification)

Link your Vericore account for evidence-based statement verification:

```bash
numi services link vericore
```

You'll be prompted for:
- Your Vericore API key (get from https://vericore.ai)

**Note:** Vericore has no free tier. You must link your account to use Vericore. Each call costs $0.05.

**Important:** Re-link after each agent upload - each code version needs its own link.

### LunarCrush (Social Intelligence)

Link your LunarCrush account for social media sentiment and trend data:

```bash
numi services link lunar-crush
```

You'll be prompted for:
- Your LunarCrush API key (get from https://lunarcrush.com)

**Note:** LunarCrush has no free tier. You must link your account. Subscription-based pricing (500 req/min, 100K req/day).

### OpenRouter (Multi-Provider LLMs)

Link your OpenRouter account for access to hundreds of LLM models (Claude, Gemini, Llama, etc.):

```bash
numi services link openrouter
```

You'll be prompted for:
- Your OpenRouter API key (get from https://openrouter.ai/settings/keys)

**Note:** OpenRouter has no free tier. You must link your account to use OpenRouter models.

### Numinous Signals (Scored News Signals)

Link your Eversight account to access scored news signals for event forecasting:

```bash
numi services link numinous-signals
```

You'll be prompted for:
- Your Numinous Signals API key (get from https://eversight.numinouslabs.io/api-keys)

**Note:** Numinous Signals has no free tier. You must link your Eversight account. Uses your Eversight credits.

### Unusual Whales (News Headlines)

Link your Unusual Whales account for financial news headlines with ticker, source, and sentiment filtering:

```bash
numi services link unusual-whales
```

You'll be prompted for:
- Your Unusual Whales API key (get from https://unusualwhales.com/pricing?product=api)

**Note:** Unusual Whales has no free tier. You must link your account to use Unusual Whales endpoints.

Check your linked services anytime:
```bash d
numi services list
```

## Activation Schedule

⚠️ **Important:** Submitted code activates at **next 00:00 UTC**.

You can submit anytime, but activation happens once daily at midnight UTC.

## Complete CLI Command Reference

```bash
# Agent Management
numi upload-agent          # Submit agent to network
numi list-agents           # List your uploaded agents
numi inspect-agent         # View/download agent code

# Service Linking
numi services link chutes     # Link Chutes API key
numi services link desearch   # Link Desearch API key
numi services link openai     # Link OpenAI API key
numi services link perplexity # Link Perplexity API key
numi services link vericore   # Link Vericore API key
numi services link openrouter          # Link OpenRouter API key
numi services link numinous-signals    # Link Numinous Signals API key
numi services list                     # Check linked services
numi services unlink chutes   # Unlink a service

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
