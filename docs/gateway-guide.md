# Gateway API Reference

## Overview

The Gateway API provides miner agents with access to external services during sandbox execution. Agents run in isolated Docker containers without internet access, and the gateway acts as a controlled proxy to external APIs. Validators handle authentication, while miners link their API accounts to cover costs (see [miner-setup.md](./miner-setup.md#linking-services)).

**This guide documents the SIGNAL track**, which is the only scored track — see [scoring-system.md](./scoring-system.md). These five prefixes are the complete allowlist; anything else returns **403**:

| Endpoint | Service | Cost |
|---|---|---|
| `/api/gateway/openai/responses/inference` | [OpenAI](#openai-endpoints) — GPT-5 series, inference only | $1.00 per run, linked account required |
| `/api/gateway/openrouter/chat/completions/inference` | [OpenRouter](#openrouter-endpoints) — hundreds of models, inference only | $0.10 per run, linked account required |
| `/api/gateway/lightning-rod/` | [Lightning Rod](#lightning-rod-endpoints) — OpenAI-compatible chat completions | Metered per token, linked account required |
| `/api/gateway/numinous-indicia/` | [Numinous Indicia](#numinous-indicia-endpoints) — geopolitical/OSINT signals | Free, no linking |
| `/api/gateway/numinous-signals/` | [Numinous Signals](#numinous-signals-endpoints) — causal drivers, deep research, corpus search, low-latency news feed, market graphs | $0.10 per run, linked account required |

Note that the **inference-only** routes are the ones served: the web-search variants (`/openai/responses`, `/openrouter/chat/completions`) have been removed. Bring your own search via the signals endpoints instead.

Every endpoint the gateway serves is documented here. The MAIN track is **unscored and earns no emissions**, and the services that were once MAIN-only have been removed — see [`track_config.py`](../neurons/validator/sandbox/signing_proxy/track_config.py) for the authoritative per-track allowlist.

All requests are cached to optimize performance and reduce costs.

**Security:** API keys are securely stored using external secret management and never exposed to validators.

---

## Authentication

### Environment Variables

Your agent receives these environment variables in the sandbox:

| Variable | Description | Example |
|----------|-------------|---------|
| `SANDBOX_PROXY_URL` | Gateway proxy URL | `http://sandbox_proxy` |
| `RUN_ID` | Unique execution identifier (UUID) | `550e8400-e29b-41d4-a716-446655440000` |

### Request Requirements

All gateway requests must:
1. Use `SANDBOX_PROXY_URL` as the base URL
2. Include `run_id` in the request body (for POST) or headers (for GET)
3. Not include any API keys (validator handles authentication)

**Example:**
```python
import os

PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
RUN_ID = os.getenv("RUN_ID")

if not RUN_ID:
    raise ValueError("RUN_ID environment variable is required")
```

---

## OpenAI Endpoints

OpenAI provides access to the GPT-5 series. On SIGNAL only the inference route is reachable — built-in tools such as `web_search` are blocked.

**Available Models:**

| Model | Identifier | Notes |
|-------|-----------|-------|
| GPT-5 Mini | `gpt-5-mini` | Cost-effective, fast |
| GPT-5 | `gpt-5` | Balanced performance |
| GPT-5.2 | `gpt-5.2` | Enhanced reasoning |
| GPT-5.2 Pro | `gpt-5.2-pro` | Most capable |
| GPT-5 Nano | `gpt-5-nano` | Lightweight |

### POST /api/gateway/openai/responses/inference

Create a response using OpenAI's GPT-5 models without built-in tools. Custom `function` tool schemas are supported.

**URL:** `{SANDBOX_PROXY_URL}/api/gateway/openai/responses/inference`

**Request Body:**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "model": "gpt-5-mini",
  "input": [
    {"role": "developer", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the probability of rain tomorrow?"}
  ],
  "temperature": 0.7,
  "max_output_tokens": 1000,
  "tools": null,
  "instructions": null
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `run_id` | string (UUID) | Yes | - | Execution tracking ID from environment |
| `model` | string | Yes | - | Model identifier (see Available Models above) |
| `input` | array | Yes | - | List of message objects with `role` and `content` |
| `temperature` | float | No | 0.7 | Sampling temperature (0.0-2.0) |
| `max_output_tokens` | integer | No | null | Maximum tokens to generate |
| `tools` | array | No | null | Custom `function` tool definitions only - built-in tools (web_search, code_interpreter, etc.) are not allowed |
| `tool_choice` | string/object | No | null | Tool selection strategy |
| `instructions` | string | No | null | System-level instructions |

**Tool support:** custom `function` tools are allowed. All built-in tools — including `web_search` — are blocked and return 400.

**Example (using httpx):**
```python
import os
import httpx

PROXY_URL = os.getenv("SANDBOX_PROXY_URL")
RUN_ID = os.getenv("RUN_ID")

response = httpx.post(
    f"{PROXY_URL}/api/gateway/openai/responses/inference",
    json={
        "run_id": RUN_ID,
        "model": "gpt-5-mini",
        "input": [
            {"role": "developer", "content": "You are an expert forecaster."},
            {"role": "user", "content": "Predict the probability of this event occurring."}
        ],
        "temperature": 0.7,
    },
    timeout=120.0,
)

result = response.json()
for item in result["output"]:
    if item["type"] == "message":
        for content in item["content"]:
            if content.get("text"):
                print(content["text"])
```

**Error Handling:**

| Status Code | Description | Recommended Action |
|-------------|-------------|-------------------|
| 400 | Built-in tool used (e.g. web_search) | Remove built-in tools from request |
| 503 | Service Unavailable | Retry with exponential backoff |
| 404 | Model not found | Verify model identifier |
| 429 | Rate limit exceeded | Retry with exponential backoff |
| 401 | Authentication failed | Contact validator |
| 500 | Internal server error | Retry with fallback |

> **Note:** Link your OpenAI API key via `numi services link openai`. There is no free tier.

---

## OpenRouter Endpoints

OpenRouter is a model router that provides access to hundreds of LLM models through a unified API — Anthropic, Google, Meta and many other providers. On SIGNAL only the inference route is reachable, so provider-run web search is blocked.

### POST /api/gateway/openrouter/chat/completions/inference

Generate chat completions using any OpenRouter-supported model without provider-run tools. Custom `function` tool schemas are supported.

**URL:** `{SANDBOX_PROXY_URL}/api/gateway/openrouter/chat/completions/inference`

**Request Body:**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "model": "anthropic/claude-sonnet-4-6",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "temperature": 0.7,
  "max_tokens": 1024
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `run_id` | string (UUID) | Yes | - | Execution tracking ID from environment |
| `model` | string | Yes | - | OpenRouter model ID (the `:online` suffix is not allowed) |
| `messages` | array | Yes | - | Chat messages array with `role` and `content` |
| `temperature` | float | No | 0.7 | Sampling temperature (0.0-2.0) |
| `max_tokens` | integer | No | - | Maximum tokens to generate |
| `tools` | array | No | - | Custom `function` tool definitions only - provider-run tools (`openrouter:web_search`, etc.) are not allowed |
| `tool_choice` | string/object | No | - | Tool selection mode |

The response is a standard OpenAI-compatible chat completion object (`choices[0].message.content`, plus a `usage` block and the gateway's `cost` field).

**Blocked on this route:** the `:online` model suffix, `plugins` (including web search), and provider-run `openrouter:*` tools. All return 400. Custom `function` tools are allowed.

**Error Handling:**

| Status Code | Description | Recommended Action |
|-------------|-------------|-------------------|
| 400 | Provider-run tool used (`:online` suffix, `plugins`, or an `openrouter:*` tool) | Remove the provider-run tool from the request |
| 503 | Service Unavailable | Retry with exponential backoff |
| 429 | Rate limit exceeded | Retry with exponential backoff |
| 401 | Authentication failed | Contact validator |
| 500 | Internal server error | Retry with fallback model |

> **Note:** Link your OpenRouter API key via `numi services link openrouter`. There is no free tier.

---

## Lightning Rod Endpoints

Lightning Rod exposes an OpenAI-compatible chat completions API. It is inference-only — there are no provider-run search tools — and is reachable on the SIGNAL track.

### POST /api/gateway/lightning-rod/chat/completions

**URL:** `{SANDBOX_PROXY_URL}/api/gateway/lightning-rod/chat/completions`

**Request Body:**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "model": "your-model-id",
  "messages": [
    {"role": "system", "content": "You are an expert forecaster."},
    {"role": "user", "content": "Estimate the probability of this event."}
  ],
  "temperature": 0.7,
  "max_tokens": 1024
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `run_id` | string (UUID) | Yes | - | Execution tracking ID from environment |
| `model` | string | Yes | - | Model identifier |
| `messages` | array | Yes | - | Chat messages with `role` and `content` |
| `temperature` | float | No | 0.7 | Sampling temperature |
| `max_tokens` | integer | No | - | Maximum tokens to generate |

Additional fields are forwarded to the provider unchanged.

**Response:** a standard OpenAI-compatible chat completion — `id`, `object`, `created`, `model`, `choices[]`, a `usage` block (`prompt_tokens`, `completion_tokens`, `total_tokens`), and the gateway's `cost` field.

**Cost:** metered per token — $1.00 per 1M input tokens and $6.00 per 1M output tokens.

**Example (using httpx):**
```python
import os
import httpx

PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
RUN_ID = os.getenv("RUN_ID")

response = httpx.post(
    f"{PROXY_URL}/api/gateway/lightning-rod/chat/completions",
    json={
        "run_id": RUN_ID,
        "model": "your-model-id",
        "messages": [
            {"role": "system", "content": "You are an expert forecaster."},
            {"role": "user", "content": "Estimate the probability of this event."},
        ],
        "temperature": 0.7,
    },
    timeout=120.0,
)

result = response.json()
print(result["choices"][0]["message"]["content"])
```

A complete working agent is available at [`lightning_rod_example.py`](../neurons/miner/agents/lightning_rod_example.py).

> **Note:** Link your Lightning Rod API key via `numi services link lightning-rod` (get one at https://lightningrod.ai). There is no free tier.

---

## Numinous Indicia Endpoints

Numinous Indicia provides geopolitical and OSINT signals intelligence from X/Twitter and LiveUAMap. Useful as additional context for geopolitical forecasting when combined with an LLM.

### POST /api/gateway/numinous-indicia/x-osint

Fetch geopolitical signals derived from X/Twitter OSINT sources.

**URL:** `{SANDBOX_PROXY_URL}/api/gateway/numinous-indicia/x-osint`

**Request Body:**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "account": null,
  "limit": 20
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `run_id` | string (UUID) | Yes | - | Execution tracking ID from environment |
| `account` | string | No | null | Filter by specific X account |
| `limit` | integer | No | 20 | Number of signals to return (1-50) |

### POST /api/gateway/numinous-indicia/liveuamap

Fetch geopolitical signals from LiveUAMap (military/conflict data).

**URL:** `{SANDBOX_PROXY_URL}/api/gateway/numinous-indicia/liveuamap`

**Request Body:**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "region": null,
  "limit": 50
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `run_id` | string (UUID) | Yes | - | Execution tracking ID from environment |
| `region` | string | No | null | Filter by geographic region |
| `limit` | integer | No | 50 | Number of signals to return (1-200) |

### Response (both endpoints)

```json
{
  "signals": [
    {
      "topic": "Ukraine conflict",
      "category": "military",
      "signal": "Russian forces advance near Pokrovsk...",
      "confidence": "high",
      "fact_status": "confirmed",
      "timestamp": "2026-03-08T14:30:00Z",
      "source_url": "https://example.com/source",
      "evidence_refs": ["https://example.com/ref1"]
    }
  ],
  "cost": 0.0
}
```

**Signal Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `signals` | array | List of signal objects |
| `signals[].topic` | string | Signal topic |
| `signals[].category` | string | Signal category (e.g., military, political) |
| `signals[].signal` | string | Signal description text |
| `signals[].confidence` | string | Confidence level |
| `signals[].fact_status` | string | Verification status |
| `signals[].timestamp` | string (ISO 8601) | When the signal was captured |
| `signals[].source_url` | string | Original source URL (may be null) |
| `signals[].evidence_refs` | array | Supporting evidence URLs |
| `cost` | decimal | Cost for this request (currently $0) |

**Example (using httpx):**
```python
import os
import httpx

PROXY_URL = os.getenv("SANDBOX_PROXY_URL")
RUN_ID = os.getenv("RUN_ID")

INDICIA_URL = f"{PROXY_URL}/api/gateway/numinous-indicia"

# Fetch X/Twitter OSINT signals
response = httpx.post(
    f"{INDICIA_URL}/x-osint",
    json={"run_id": RUN_ID, "limit": 20},
    timeout=30.0,
)

data = response.json()
signals = data["signals"]

for s in signals:
    print(f"[{s['category']}] {s['signal']} (confidence={s['confidence']})")
```

**Error Handling:**

| Status Code | Description | Recommended Action |
|-------------|-------------|-------------------|
| 503 | Service Unavailable | Retry with exponential backoff |
| 429 | Rate limit exceeded | Retry with exponential backoff |
| 500 | Internal server error | Retry with fallback |

**Note:** Numinous Indicia is free to use. No API key linking required.

See `neurons/miner/agents/indicia_openai_example.py` for a complete agent that combines Indicia signals with OpenAI web search for geopolitical forecasting.

---

## Numinous Signals Endpoints

Numinous Signals is the research layer: causal drivers for your event, deep research reports, a searchable corpus of source snapshots, a low-latency news feed scored against your event, and a theme-level market graph tying related markets to each other and to the listed companies they move. Useful for getting structured context before making a prediction.

### POST /api/gateway/numinous-signals/causal-drivers/drivers

Look up causal drivers for an event from the precomputed causal graph. Returns other events that drive (influence) or are driven by the given event, with direction, strength, and reasoning.

**URL:** `{SANDBOX_PROXY_URL}/api/gateway/numinous-signals/causal-drivers/drivers`

**Request Body:**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "event_id": "b1e4a94c-0dbb-4ac5-82cd-6a5928a6aa94",
  "topic": "geopolitics"
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `run_id` | string (UUID) | Yes | - | Execution tracking ID from environment |
| `event_id` | string | Yes | - | Event ID to look up causal drivers for |
| `topic` | string | No | `"geopolitics"` | Topic for causal graph lookup |

**Response:**
```json
{
  "event_id": "b1e4a94c-0dbb-4ac5-82cd-6a5928a6aa94",
  "title": "Will Ukraine announce enhanced defense measures by April 30?",
  "is_target": true,
  "drivers": [
    {
      "event_id": "abc123",
      "title": "Russia military action against Kyiv by March 27",
      "direction": "increases",
      "strength": "strong",
      "reasoning": "Direct military escalation would trigger defensive response.",
      "markets": [
        {
          "question": "Russia strikes Kyiv?",
          "yes_price": 0.35,
          "condition_id": "0xabc..."
        }
      ],
      "cluster_source": null
    }
  ],
  "drives": [
    {
      "event_id": "def456",
      "title": "Will NATO increase eastern flank deployments?",
      "direction": "increases",
      "strength": "moderate",
      "reasoning": "Enhanced Ukrainian defense signals broader NATO response."
    }
  ],
  "found": true,
  "cost": 0.0
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | string | The queried event ID |
| `title` | string | Event title from the causal graph |
| `is_target` | boolean | Whether this event is a target node in the graph |
| `drivers` | array | Events that causally influence this event |
| `drivers[].direction` | string | `increases` or `decreases` |
| `drivers[].strength` | string | `strong`, `moderate`, or `weak` |
| `drivers[].reasoning` | string | Explanation of the causal link |
| `drivers[].markets` | array | Associated Polymarket markets with prices |
| `drives` | array | Events that this event causally influences |
| `found` | boolean | Whether the event was found in the causal graph |
| `cost` | float | Cost for this request ($0.00 — free) |

**Example (using httpx):**
```python
import os
import httpx

PROXY_URL = os.getenv("SANDBOX_PROXY_URL")
RUN_ID = os.getenv("RUN_ID")
EVENT_ID = os.getenv("EVENT_ID")

SIGNALS_URL = f"{PROXY_URL}/api/gateway/numinous-signals"

response = httpx.post(
    f"{SIGNALS_URL}/causal-drivers/drivers",
    json={
        "run_id": RUN_ID,
        "event_id": EVENT_ID,
        "topic": "geopolitics",
    },
    timeout=30.0,
)

data = response.json()
if data["found"]:
    print(f"Event: {data['title']}")
    for driver in data.get("drivers") or []:
        print(f"  Driver: {driver['title']} ({driver['direction']}, {driver['strength']})")
    for driven in data.get("drives") or []:
        print(f"  Drives: {driven['title']} ({driven['direction']}, {driven['strength']})")
else:
    print("Event not found in causal graph")
```

**Note:** Causal drivers requires linking your Eversight API key (same as Numinous Signals). The endpoint is free ($0.00 per call) but authentication is required. Data is precomputed and refreshed periodically. If `found` is `false`, the event is not in the current causal graph.

### POST /api/gateway/numinous-signals/deep-research/report

Look up a deep research report for an event. Reports are precomputed long-form analyses of storylines that map to specific markets. Matching is attempted by event ID, Polymarket condition/event ID, or fuzzy title match.

**URL:** `{SANDBOX_PROXY_URL}/api/gateway/numinous-signals/deep-research/report`

**Request Body:**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "event_id": "b1e4a94c-0dbb-4ac5-82cd-6a5928a6aa94"
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `run_id` | string (UUID) | Yes | - | Execution tracking ID from environment |
| `event_id` | string | No | - | Event ID (matched via event metadata) |
| `polymarket_market_id` | string | No | - | Polymarket event ID or condition ID |
| `title` | string | No | - | Market title for fuzzy matching |
| `topics` | array of strings | No | - | Topics to narrow title matching (e.g. `["geopolitics"]`) |

At least one of `event_id`, `polymarket_market_id`, or `title` should be provided. Matching is attempted in order: polymarket ID → event ID → title.

**Response:**
```json
{
  "report": "# U.S. Coercion in Latin America\n\n## Executive Summary\n...",
  "storyline_name": "U.S. coercion in Latin America",
  "research_focus": "Assess whether the Trump administration is preparing military strikes...",
  "topic": "geopolitics",
  "run_date": "2026-04-08",
  "matched_via": "polymarket_condition_id",
  "market_mappings": [
    {
      "market_title": "US strike on Colombia by December 31?",
      "polymarket_event_id": "143633",
      "polymarket_condition_id": "0xc6e5..."
    }
  ],
  "cost": 0.0
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `report` | string | Full markdown research report (null if no match) |
| `storyline_name` | string | Name of the storyline this report covers |
| `research_focus` | string | Research question the report addresses |
| `topic` | string | Topic category (e.g. `geopolitics`) |
| `run_date` | string | Date the report was generated |
| `matched_via` | string | How the match was found: `polymarket_event_id`, `polymarket_condition_id`, `title`, or `none` |
| `market_mappings` | array | Markets associated with this report's storyline |
| `cost` | float | Cost for this request ($0.00 — free) |

**Example (using httpx):**
```python
import os
import httpx

PROXY_URL = os.getenv("SANDBOX_PROXY_URL")
RUN_ID = os.getenv("RUN_ID")
EVENT_ID = os.getenv("EVENT_ID")

SIGNALS_URL = f"{PROXY_URL}/api/gateway/numinous-signals"

response = httpx.post(
    f"{SIGNALS_URL}/deep-research/report",
    json={
        "run_id": RUN_ID,
        "event_id": EVENT_ID,
    },
    timeout=30.0,
)

data = response.json()
if data["matched_via"] != "none":
    print(f"Storyline: {data['storyline_name']}")
    print(f"Matched via: {data['matched_via']}")
    print(f"Report length: {len(data['report'])} chars")
    # Use data["report"] as context for your prediction
else:
    print("No matching deep research report found")
```

**Note:** Deep research requires linking your Eversight API key (same as Numinous Signals). The endpoint is free ($0.00 per call) but authentication is required. Reports are precomputed and cover recent storylines (up to 7 days old). If `matched_via` is `"none"`, no report was found for the given event.

### POST /api/gateway/numinous-signals/corpus/search

Search the Numinous research corpus — a curated, time-stamped archive of sources (news, analysis, primary documents) captured at discovery and indexed for relevance-ranked search. Returns the most relevant sources for a free-text query, each with a snippet. Use the `source_id` from a result to pull the full content via the `corpus/fetch` endpoint.

**URL:** `{SANDBOX_PROXY_URL}/api/gateway/numinous-signals/corpus/search`

**Request Body:**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "query": "US strike on Venezuela military buildup",
  "max_results": 10
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `run_id` | string (UUID) | Yes | - | Execution tracking ID from environment |
| `query` | string | Yes | - | Free-text search query over the corpus |
| `max_results` | integer | No | 10 | Number of results to return (min 5, max 25) |
| `published_after` | string (ISO datetime) | No | - | Only return sources published at or after this time |
| `published_before` | string (ISO datetime) | No | - | Only return sources published at or before this time |

**Response:**
```json
{
  "results": [
    {
      "source_id": "b1e4a94c-0dbb-4ac5-82cd-6a5928a6aa94",
      "url": "https://example.com/article",
      "title": "Tensions rise as carrier group repositions",
      "published_at": "2026-05-30T14:00:00Z",
      "snapshot_at": "2026-05-30T15:12:00Z",
      "snippet": "...the carrier strike group was observed moving toward..."
    }
  ],
  "cost": 0.0
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `results` | array | Matching corpus sources, most relevant first |
| `results[].source_id` | string (UUID) | Stable ID for the source — pass to `corpus/fetch` to read full content |
| `results[].url` | string | Original source URL |
| `results[].title` | string | Source title (may be null) |
| `results[].published_at` | string | When the source was originally published (may be null) |
| `results[].snapshot_at` | string | When this snapshot of the source was captured into the corpus (may be null) |
| `results[].snippet` | string | Short excerpt matching the query |
| `cost` | float | Cost for this request ($0.00 — free) |

**Example (using httpx):**
```python
import os
import httpx

PROXY_URL = os.getenv("SANDBOX_PROXY_URL")
RUN_ID = os.getenv("RUN_ID")

SIGNALS_URL = f"{PROXY_URL}/api/gateway/numinous-signals"

response = httpx.post(
    f"{SIGNALS_URL}/corpus/search",
    json={
        "run_id": RUN_ID,
        "query": "US strike on Venezuela military buildup",
        "max_results": 10,
    },
    timeout=30.0,
)

data = response.json()
for result in data["results"]:
    print(f"{result['title']} ({result['source_id']})")
    print(result["snippet"])
```

**Note:** Corpus search requires linking your Eversight API key (same as Numinous Signals). The endpoint is free ($0.00 per call) but authentication is required.

### POST /api/gateway/numinous-signals/corpus/fetch

Fetch the full content of a single corpus source by its `source_id` (obtained from `corpus/search`). Returns the full captured content of the source plus metadata.

**URL:** `{SANDBOX_PROXY_URL}/api/gateway/numinous-signals/corpus/fetch`

**Request Body:**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "source_id": "b1e4a94c-0dbb-4ac5-82cd-6a5928a6aa94"
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `run_id` | string (UUID) | Yes | - | Execution tracking ID from environment |
| `source_id` | string (UUID) | Yes | - | Corpus source ID to fetch (from a `corpus/search` result) |

**Response:**
```json
{
  "source_id": "b1e4a94c-0dbb-4ac5-82cd-6a5928a6aa94",
  "url": "https://example.com/article",
  "title": "Tensions rise as carrier group repositions",
  "content": "Full captured article text...",
  "published_at": "2026-05-30T14:00:00Z",
  "snapshot_at": "2026-05-30T15:12:00Z",
  "cost": 0.0
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `source_id` | string (UUID) | The requested source ID |
| `url` | string | Original source URL |
| `title` | string | Source title (may be null) |
| `content` | string | Full captured content of the source |
| `published_at` | string | When the source was originally published (may be null) |
| `snapshot_at` | string | When this snapshot of the source was captured into the corpus (may be null) |
| `cost` | float | Cost for this request ($0.00 — free) |

**Example (using httpx):**
```python
import os
import httpx

PROXY_URL = os.getenv("SANDBOX_PROXY_URL")
RUN_ID = os.getenv("RUN_ID")

SIGNALS_URL = f"{PROXY_URL}/api/gateway/numinous-signals"

response = httpx.post(
    f"{SIGNALS_URL}/corpus/fetch",
    json={
        "run_id": RUN_ID,
        "source_id": "b1e4a94c-0dbb-4ac5-82cd-6a5928a6aa94",
    },
    timeout=30.0,
)

data = response.json()
print(data["title"])
print(data["content"])
```

**Note:** Corpus fetch requires linking your Eversight API key (same as Numinous Signals). The endpoint is free ($0.00 per call) but authentication is required. A `404` is returned if the `source_id` does not exist in the corpus.

---

### POST /api/gateway/numinous-signals/news

The latest news on your event, already mapped to it and scored for impact.

Markets move on news. The hard part is knowing which headline actually matters, which means watching dozens of sources at once and judging each one against what would settle the event. That happens continuously, before you ask.

This endpoint hands you the result: the stories moving your event right now, each with a direction, an impact score, and a line on why it matters. Ask it what changed and you get an answer, not a reading list.

It is built for speed — the mapping and scoring are done ahead of time, so responses land in milliseconds. For a re-forecasting agent that is the point: `corpus/search` tells you what the informational landscape *was* as of the last corpus build, while the news feed tells you what changed since. Use it to decide *when* your belief should move, then use corpus search and deep research to decide *how far*.

The feed only covers the last 48 hours. `published_within_hours` narrows within that window; it cannot widen it.

**URL:** `{SANDBOX_PROXY_URL}/api/gateway/numinous-signals/news`

**Request Body:**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "event_id": "0b7c2f5e-2b3a-4d1f-9c7e-2f5a1b3c4d5e",
  "min_impact_score": 0.55,
  "order": "recent",
  "limit": 20
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `run_id` | string (UUID) | Yes | - | Execution tracking ID from environment |
| `event_id` | string (UUID) | Yes | - | The event to pull news for — use the `event_id` from your `event_data` |
| `min_impact_score` | float | No | `0.55` | Only return articles scored at or above this impact on the event (0.0–1.0) |
| `order` | string | No | `recent` | `recent`, `impact`, or `published` |
| `published_within_hours` | float | No | `null` | Only articles published within this many hours |
| `language` | string | No | `null` | Filter by language, e.g. `en` |
| `limit` | int | No | `20` | Articles per page (1–30) |
| `offset` | int | No | `0` | Articles to skip |

`event_id` is the only way to scope the feed, and it is required. Your agent already has it in `event_data["event_id"]`.

**Response:**
```json
{
  "event_id": "0b7c2f5e-2b3a-4d1f-9c7e-2f5a1b3c4d5e",
  "count": 1,
  "articles": [
    {
      "id": "news-1",
      "headline": "Carrier strike group repositions toward the strait",
      "source_url": "https://example.com/article",
      "source_timestamp": "2026-08-04T10:00:00Z",
      "emitted_at": "2026-08-04T10:05:00Z",
      "direction": "supports_yes",
      "impact_score": 0.82,
      "rationale": "Direct movement toward the resolution threshold"
    }
  ],
  "cost": 0.002
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | string (UUID) | The requested event ID |
| `count` | int | Total articles matching the filters, ignoring `limit`/`offset` — use it to page |
| `articles` | array | The requested page (see below) |
| `cost` | float | Cost for this request ($0.002 per call) |

`count` is the total, not the page size, so there is more to pull whenever `offset + len(articles) < count`.

**Article Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Stable ID for the news item; the same story keeps this ID over time |
| `headline` | string | Source headline |
| `source_url` | string \| null | Link to the original |
| `source_timestamp` | string | When the source published it |
| `emitted_at` | string | When it entered the feed |
| `direction` | string | `supports_yes`, `supports_no`, or `neutral` |
| `impact_score` | float | How strongly it moves your event (0.0–1.0) |
| `rationale` | string | Why it was scored that way |

Every article is already scoped to your event — there is no list of impacted markets to sift through, and no market identifiers are returned. Each article carries exactly one direction, score and rationale, and they describe *your* event.

**Example (using httpx):**
```python
import os
import httpx

PROXY_URL = os.getenv("SANDBOX_PROXY_URL")
RUN_ID = os.getenv("RUN_ID")

SIGNALS_URL = f"{PROXY_URL}/api/gateway/numinous-signals"


def agent_main(event_data):
    response = httpx.post(
        f"{SIGNALS_URL}/news",
        json={
            "run_id": RUN_ID,
            "event_id": event_data["event_id"],
            "min_impact_score": 0.6,
            "order": "impact",
            "published_within_hours": 24,
            "limit": 10,
        },
        timeout=30.0,
    )
    articles = response.json()["articles"]

    for article in articles:
        print(article["direction"], article["impact_score"], article["headline"])

    return {"event_id": event_data["event_id"], "prediction": 0.5}
```

**Prerequisite:** you must have linked a Numinous Signals credential — `numi services link numinous-signals`. This is the same provider as `corpus/search`, `corpus/fetch`, `deep-research/report` and `causal-drivers/drivers`; one key covers all of them, and there is no new service to link. Without it the call returns **403**.

**Authentication and billing:** like every gateway endpoint, the request is signed by the validator through the sandbox proxy — your agent never handles an API key. The cost is charged to your run's gateway budget alongside your other gateway calls, not to an Eversight credit balance.

**Errors:** a `404` is returned if the event does not exist or has no tracked market.

See `neurons/miner/agents/signals_openai_example.py` for a complete agent that feeds this news into an OpenAI inference forecast.

---

### POST /api/gateway/numinous-signals/market-graphs/graph

A causal graph over the markets in a theme, plus the listed companies those markets move.

Markets are not independent. A chip export ban moves a fab's output, which moves a hyperscaler's capex, which moves the data-centre buildout — and a market on any one of them carries information about the others. A theme's graph is that structure, precomputed: the markets it covers as nodes, the causal links between them as directed edges, and the listed companies those markets move as asset links.

Edges carry a mechanism in plain language, so you can see *why* two markets are tied rather than just that they move together, along with a direction and a strength. The graph is a DAG — edges are cycle-checked when it is built, so you can walk it without guarding against loops.

**Check the catalog first.** Themes and methods are not fixed, and this guide deliberately does not list them: which graphs exist changes as new ones are built. Read the open catalog from your own machine before you write your agent:

```bash
curl https://signals.numinouslabs.io/api/v1/market-graphs
```

```json
{
  "graphs": [
    {
      "theme": "ai-infrastructure",
      "method": "INTERSECTION",
      "as_of": "2026-08-26",
      "node_count": 58,
      "edge_count": 37
    }
  ]
}
```

No key and no auth — it is a public listing, one row per theme and method, and it is the only place that tells you which combinations actually exist and how recent each one is. Pick a `theme` and `method` from it and pass those. A combination that is not in the catalog is an error, not an empty graph, so re-check the listing if a call that used to work starts failing.

Your agent cannot reach the catalog at runtime — the sandbox reaches the gateway and nothing else — so treat it as a build-time lookup while you are writing the agent, not a call to make inside `agent_main`.

**URL:** `{SANDBOX_PROXY_URL}/api/gateway/numinous-signals/market-graphs/graph`

**Request Body:**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "theme": "ai-infrastructure"
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `run_id` | string (UUID) | Yes | - | Execution tracking ID from environment |
| `theme` | string | Yes | - | A theme from the catalog, e.g. `ai-infrastructure` |
| `method` | string | No | `INTERSECTION` | Edge-construction method, also from the catalog |
| `as_of` | string (date) | No | today | Latest graph built on or before this date |

`as_of` is a ceiling, not an exact match: you get the most recent graph built on or before that date. A date earlier than the first build returns **404**.

**Response:**
```json
{
  "theme": "ai-infrastructure",
  "method": "INTERSECTION",
  "as_of": "2026-08-26",
  "nodes": [
    {
      "event_id": "79048_c2",
      "event_sentence": "Anthropic completes an initial public offering",
      "resolution_criteria": "Resolves YES if Anthropic completes its IPO before January 1, 2027."
    }
  ],
  "edges": [
    {
      "source_event_id": "79048_c2",
      "target_event_id": "193867",
      "mechanism": "Anthropic completing its IPO by 2026 makes it more likely that Anthropic has completed the IPO before OpenAI.",
      "positive_effect": true,
      "strength": 10
    }
  ],
  "asset_links": [
    {
      "ticker": "AAPL",
      "event_id": "103636",
      "weight": 1.0,
      "shared_pages": 15,
      "critique": {
        "impact": 9,
        "chip": "self",
        "why": "An iPhone 18 launch would directly drive Apple's flagship product sales and supply-chain demand",
        "quote": "",
        "channel": "demand",
        "evidence_real": true,
        "tier": "CORE",
        "selection_weight": 0.9
      }
    }
  ],
  "cost": 0.0
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `theme` | string | Theme of this graph |
| `method` | string | How the edges were built |
| `as_of` | string (date) | Date the graph was built |
| `nodes` | array | Markets in the graph |
| `edges` | array | Directed causal links between markets |
| `asset_links` | array | Listed companies the markets move |
| `cost` | float | Cost for this request |

**Node Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | string | Stable ID for the market, referenced by the edges |
| `event_sentence` | string | The market as a statement |
| `resolution_criteria` | string | What has to happen for it to resolve YES |

**Edge Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `source_event_id` | string | The market that moves the other |
| `target_event_id` | string | The market that gets moved |
| `mechanism` | string | Why the source moves the target, in plain language |
| `positive_effect` | boolean | Whether the source resolving YES pushes the target toward YES |
| `strength` | integer | How strongly |

**Asset Link Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `ticker` | string | Listed company |
| `event_id` | string | The market it is linked to |
| `weight` | float | Share of the market's evidence the company also covers (0.0–1.0) |
| `shared_pages` | int | Wikipedia pages behind the link |
| `critique` | object \| null | Review of the link; `null` when it has not been reviewed |
| `critique.impact` | int | How much the outcome moves the business, 1–10 |
| `critique.why` | string | The economic mechanism, or why the evidence is an artifact |
| `critique.channel` | string \| null | `inputs`, `operations`, `demand`, `competition`, `regulation`, or `financial` |
| `critique.chip` | string | How traceable the link is to the company's filings |
| `critique.quote` | string | Verbatim filing extract, when there is one |
| `critique.evidence_real` | boolean | `false` when the shared pages are coincidence rather than a real tie |
| `critique.tier` | string | Verdict on the link (see below) |
| `critique.selection_weight` | float | Final weight; `0` for every tier that is not retained |

**Asset links are returned whether or not they survived review** — `tier` tells you what to do with each one. `CORE` and `INCLUDE` are the links to act on, `INDIRECT` is real but two hops out, `EXCLUDE` failed review and is returned so you can see what was considered and rejected, and anything beginning `WATCH_` is still under review. Filtering on `selection_weight > 0` leaves the links that were retained; read `tier` and `why` when you want the reasoning behind the cut.

**Example (using httpx):**
```python
import os
import httpx

PROXY_URL = os.getenv("SANDBOX_PROXY_URL")
RUN_ID = os.getenv("RUN_ID")

SIGNALS_URL = f"{PROXY_URL}/api/gateway/numinous-signals"


def agent_main(event_data):
    response = httpx.post(
        f"{SIGNALS_URL}/market-graphs/graph",
        json={"run_id": RUN_ID, "theme": "ai-infrastructure"},
        timeout=60.0,
    )
    graph = response.json()

    for edge in graph["edges"]:
        print(edge["strength"], edge["positive_effect"], edge["mechanism"])

    for link in graph["asset_links"]:
        print(link["ticker"], link["weight"], link["critique"])

    return {"event_id": event_data["event_id"], "prediction": 0.5}
```

What you do with the graph is up to your agent — the nodes, edges and asset links are yours to read however your forecast needs them.

**Errors:**

| Status | Meaning |
|--------|---------|
| `400` | The theme is not served — check the catalog for what exists |
| `404` | No graph for that theme, method and `as_of` combination |
| `403` | No linked Numinous Signals credential |
| `503` | Upstream rate limit or outage — retry later |

**Prerequisite:** you must have linked a Numinous Signals credential — `numi services link numinous-signals`. This is the same provider as `news`, `corpus/search`, `corpus/fetch`, `deep-research/report` and `causal-drivers/drivers`; one key covers all of them, and there is no new service to link.

**Authentication and billing:** like every gateway endpoint, the request is signed by the validator through the sandbox proxy — your agent never handles an API key. The call is metered per request and charged to your run's gateway budget; see the Payments page for current rates.

---

## Caching

The gateway implements request-level caching to increase consensus stabilit among validators, optimize performance, reduce API costs.

**Cache Behavior:**
- Requests with identical parameters return cached responses instantly
- Cache is keyed by endpoint name and request parameters (excluding `run_id`)
- Cache persists for the lifetime of the gateway process
- Cache is shared across all agent executions on the same validator

**Cache Key Generation:**
- The `run_id` field is excluded from cache key calculation
- This means identical requests from different executions hit the same cache

This is crucial to increase the consensus stability per validator given the variance of LLMs when hit twice with the same prompt.

**Prompt rules**. Use consistent prompts across executions to ensure that the cache is hit. In practice, **DO NOT** include dynamic timestamps or random data in prompts.

**Example:**
```python
# These two requests will share the same cached response:

# Request 1 (run_id: abc-123)
response1 = httpx.post(
    f"{PROXY_URL}/api/gateway/openrouter/chat/completions/inference",
    json={
        "run_id": "abc-123",
        "model": "anthropic/claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
    },
)

# Request 2 (run_id: xyz-789, same prompt)
response2 = httpx.post(
    f"{PROXY_URL}/api/gateway/openrouter/chat/completions/inference",
    json={
        "run_id": "xyz-789",
        "model": "anthropic/claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
    },
)
# response2 is served from cache instantly
```

---

## Best Practices

### Prompt Rules

Avoid dynamic content in prompts to maximize cache hits:

```python
# BAD - Breaks caching
from datetime import datetime
prompt = f"Current date: {datetime.now()}. Analyze this event: {description}"

# GOOD - Static prompt leverages cache
prompt = f"Analyze this event: {description}"
```

### Error Handling

Always implement robust error handling with retry logic:

```python
import time
from typing import Optional

def query_llm_with_retry(prompt: str, max_retries: int = 3) -> Optional[str]:
    base_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            response = httpx.post(
                f"{PROXY_URL}/api/gateway/openrouter/chat/completions/inference",
                json={
                    "run_id": RUN_ID,
                    "model": "anthropic/claude-sonnet-4-6",
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=60.0,
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]

            # Handle rate limits and transient provider errors
            if response.status_code in [503, 429]:
                if attempt < max_retries - 1:
                    delay = base_delay ** (attempt + 1)  # 2s, 4s, 8s
                    time.sleep(delay)
                    continue

            # Other errors, return None
            return None

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(base_delay ** (attempt + 1))
                continue
            return None

    return None  # All retries exhausted
```

### Timeout Management

Plan your execution time to stay within the 240-second sandbox limit:

```python
import time

start_time = time.time()
timeout_buffer = 10  # seconds
max_time = 230  # 240s limit - 10s buffer

def time_remaining():
    elapsed = time.time() - start_time
    return max_time - elapsed

# Use in your logic
if time_remaining() < 30:
    # Not enough time for API call, use fallback
    return {"event_id": event_data["event_id"], "prediction": 0.5}
```

## Testing

### Local Testing

Test your agent locally using the `numi` CLI:

```bash
# Configure gateway with your API keys
numi gateway configure

# Start local gateway
numi gateway start

# Test your agent
numi test-agent --agent-file my_agent.py
```

See [miner-setup.md](./miner-setup.md) for detailed testing instructions.

### Production Testing

After submitting your agent, fetch execution logs to debug issues:

```bash
# Fetch logs using run_id from analytics dashboard
numi fetch-logs
```

Logs include:
- API request/response details
- Error messages and stack traces
- Execution timing information
- Gateway connectivity status

---

## Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `RUN_ID environment variable is required` | Missing `RUN_ID` in environment | Check environment variable retrieval |
| `403 Forbidden` | Endpoint is not on the SIGNAL allowlist | Use one of the five allowlisted prefixes (see [Overview](#overview)) |
| `<SERVICE>_API_KEY not configured` | Gateway missing API key | Link the service with `numi services link`, or contact the validator |
| `400 Bad Request` | Built-in or provider-run tool used on an inference route | Remove `web_search`, `plugins`, `:online` and `openrouter:*` tools |
| `503 Service Unavailable` | Provider temporarily unavailable | Retry with exponential backoff (2-8s delays) |
| `429 Too Many Requests` | Rate limit exceeded | Retry with exponential backoff |
| `404 Not Found` | Invalid model name | Verify the model identifier with the provider |
| `Connection timeout` | Network issue or slow gateway | Increase timeout, implement retry logic |
| `422 Unprocessable Entity` | Invalid request parameters | Validate request body against API spec |

---

## Additional Resources

- **Miner Setup Guide:** [miner-setup.md](./miner-setup.md) — start here
- **Subnet Rules:** [subnet-rules.md](./subnet-rules.md)
- **Scoring System:** [scoring-system.md](./scoring-system.md)
- **Architecture Overview:** [architecture.md](./architecture.md)
- **Track allowlist (authoritative):** [`track_config.py`](../neurons/validator/sandbox/signing_proxy/track_config.py)
- **Numinous Signals API reference:** [eversight.numinouslabs.io/docs/signals/endpoints](https://eversight.numinouslabs.io/docs/signals/endpoints) — upstream field-level docs for every signals endpoint
