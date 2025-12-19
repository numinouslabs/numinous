# Gateway API Reference

## Overview

The Gateway API provides miner agents with access to external AI services during sandbox execution. Agents run in isolated Docker containers without internet access, and the gateway acts as a controlled proxy to external APIs. Validators handle authentication, while miners cover Desearch API costs through their linked accounts (Chutes AI costs currently covered by the subnet).

**Available Services:**
- **Chutes AI**: LLM inference with multiple open-source models
- **Desearch AI**: Web search, social media search, and content crawling

All requests are cached to optimize performance and reduce costs.

Every sandbox run has a cost limit of \$**0.02**.

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

## Chutes AI Endpoints

Chutes AI provides access to open-source LLM models for inference.

### POST /api/gateway/chutes/chat/completions

OpenAI-compatible chat completion endpoint.

**URL:** `{SANDBOX_PROXY_URL}/api/gateway/chutes/chat/completions`

**Request Body:**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "model": "deepseek-ai/DeepSeek-V3-0324",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "temperature": 0.7,
  "max_tokens": 1000,
  "tools": null,
  "tool_choice": null
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `run_id` | string (UUID) | Yes | - | Execution tracking ID from environment |
| `model` | string | Yes | - | Model identifier (see Available Models below) |
| `messages` | array | Yes | - | List of message objects with `role` and `content` |
| `temperature` | float | No | 0.7 | Sampling temperature (0.0-2.0) |
| `max_tokens` | integer | No | null | Maximum tokens to generate |
| `tools` | array | No | null | Tool definitions for function calling |
| `tool_choice` | string/object | No | null | Tool selection strategy (`auto`, `required`, or specific tool) |

**Response:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "deepseek-ai/DeepSeek-V3-0324",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 28,
    "completion_tokens": 8,
    "total_tokens": 36
  }
}
```

**Example (using LangChain):**
```python
import os
from langchain_openai import ChatOpenAI

PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
RUN_ID = os.getenv("RUN_ID")

llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3-0324",
    base_url=f"{PROXY_URL}/api/gateway/chutes",
    api_key="not-needed",  # Gateway handles authentication
    extra_body={"run_id": RUN_ID},
)

response = llm.invoke("What is 2+2?")
print(response.content)
```

**Example (using httpx):**
```python
import os
import httpx

PROXY_URL = os.getenv("SANDBOX_PROXY_URL")
RUN_ID = os.getenv("RUN_ID")

response = httpx.post(
    f"{PROXY_URL}/api/gateway/chutes/chat/completions",
    json={
        "run_id": RUN_ID,
        "model": "deepseek-ai/DeepSeek-V3-0324",
        "messages": [{"role": "user", "content": "Hello!"}],
        "temperature": 0.7,
    },
    timeout=60.0,
)

result = response.json()
content = result["choices"][0]["message"]["content"]
```

**Available Models:**

| Model | Identifier | Notes |
|-------|-----------|-------|
| DeepSeek R1 | `deepseek-ai/DeepSeek-R1` | Latest reasoning model |
| DeepSeek R1 0528 | `deepseek-ai/DeepSeek-R1-0528` | Version-specific |
| DeepSeek V3 0324 | `deepseek-ai/DeepSeek-V3-0324` | Fast and efficient |
| DeepSeek V3.1 | `deepseek-ai/DeepSeek-V3.1` | Improved version |
| DeepSeek V3.2 Exp | `deepseek-ai/DeepSeek-V3.2-Exp` | Experimental |
| Gemma 3 4B | `unsloth/gemma-3-4b-it` | Lightweight model |
| Gemma 3 12B | `unsloth/gemma-3-12b-it` | Mid-size model |
| Gemma 3 27B | `unsloth/gemma-3-27b-it` | Larger model |
| GLM 4.5 | `zai-org/GLM-4.5` | Multilingual model |
| GLM 4.6 | `zai-org/GLM-4.6` | Latest GLM version |
| Qwen3 32B | `Qwen/Qwen3-32B` | High-performance model |
| Qwen3 235B | `Qwen/Qwen3-235B-A22B` | Large-scale model |
| Mistral Small 24B | `unsloth/Mistral-Small-24B-Instruct-2501` | Efficient instruction model |
| GPT OSS 20B | `openai/gpt-oss-20b` | Open-source GPT variant |
| GPT OSS 120B | `openai/gpt-oss-120b` | Large open-source GPT |

**Note:** Model availability can change. Check https://chutes.ai/app for the latest list of active models.

**Error Handling:**

| Status Code | Description | Recommended Action |
|-------------|-------------|-------------------|
| 503 | Service Unavailable (cold model) | Implement exponential backoff, retry after 2-8s |
| 404 | Model not found | Verify model name at https://chutes.ai/app |
| 429 | Rate limit exceeded | Implement exponential backoff |
| 401 | Authentication failed | Contact validator (gateway misconfigured) |
| 500 | Internal server error | Retry with fallback to baseline prediction |

### GET /api/gateway/chutes/status

Get real-time status and utilization metrics for all Chutes models.

**URL:** `{SANDBOX_PROXY_URL}/api/gateway/chutes/status`

**Request:**
```python
import httpx

response = httpx.get(
    f"{PROXY_URL}/api/gateway/chutes/status",
    timeout=10.0,
)
status_list = response.json()
```

**Response:**
```json
[
  {
    "chute_id": "chute-123",
    "name": "deepseek-ai/DeepSeek-R1",
    "timestamp": "2025-11-13T12:00:00Z",
    "utilization_current": 0.85,
    "utilization_5m": 0.75,
    "utilization_15m": 0.70,
    "utilization_1h": 0.65,
    "rate_limit_ratio_5m": 0.1,
    "rate_limit_ratio_15m": 0.08,
    "rate_limit_ratio_1h": 0.05,
    "total_requests_5m": 100.0,
    "completed_requests_5m": 90.0,
    "rate_limited_requests_5m": 10.0,
    "instance_count": 5,
    "action_taken": "scale_up",
    "scalable": true
  }
]
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `chute_id` | string | Unique chute identifier |
| `name` | string | Model name |
| `utilization_current` | float | Current utilization (0.0-1.0) |
| `utilization_5m` | float | 5-minute average utilization |
| `utilization_15m` | float | 15-minute average utilization |
| `utilization_1h` | float | 1-hour average utilization |
| `rate_limit_ratio_5m` | float | Ratio of rate-limited requests (5min) |
| `instance_count` | integer | Active instances |
| `action_taken` | string | Latest scaling action (`scale_up`, `scale_down`, `none`) |
| `scalable` | boolean | Whether model can scale |

**Use Case:**

Use this endpoint to select the most available model before making inference requests:

```python
import httpx

def select_best_model():
    response = httpx.get(f"{PROXY_URL}/api/gateway/chutes/status", timeout=10.0)
    status_list = response.json()

    # Filter for low utilization and low rate limiting
    available_models = [
        s for s in status_list
        if s["utilization_current"] < 0.5 and s["rate_limit_ratio_5m"] < 0.1
    ]

    if available_models:
        # Pick the least utilized model
        best = min(available_models, key=lambda x: x["utilization_current"])
        return best["name"]

    # Fallback to default
    return "deepseek-ai/DeepSeek-V3-0324"
```

---

## Desearch AI Endpoints

Desearch AI provides web search, social media search, and content crawling capabilities.

### POST /api/gateway/desearch/ai/search

AI-powered search with automatic summarization and multiple tool support.

**URL:** `{SANDBOX_PROXY_URL}/api/gateway/desearch/ai/search`

**Request Body:**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "prompt": "Latest developments in quantum computing",
  "model": "NOVA",
  "tools": ["web", "arxiv"],
  "date_filter": "PAST_WEEK",
  "result_type": "LINKS_WITH_FINAL_SUMMARY",
  "system_message": null,
  "count": 10
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `run_id` | string (UUID) | Yes | - | Execution tracking ID |
| `prompt` | string | Yes | - | Search query or question |
| `model` | string | No | `NOVA` | AI model (`NOVA`, `ORBIT`, `HORIZON`) |
| `tools` | array[string] | No | `["web"]` | Search tools to use (see Available Tools) |
| `date_filter` | string | No | null | Time range filter (see Date Filters) |
| `result_type` | string | No | null | Output format (see Result Types) |
| `system_message` | string | No | null | Custom system prompt for AI |
| `count` | integer | No | 10 | Number of results (1-100) |

**Available Tools:**

| Tool | Description |
|------|-------------|
| `web` | General web search |
| `twitter` | Twitter/X search |
| `reddit` | Reddit search |
| `hackernews` | Hacker News search |
| `wikipedia` | Wikipedia search |
| `youtube` | YouTube search |
| `arxiv` | Academic papers (arXiv) |

**Date Filters:**

| Value | Description |
|-------|-------------|
| `PAST_24_HOURS` | Last 24 hours |
| `PAST_2_DAYS` | Last 2 days |
| `PAST_WEEK` | Last 7 days |
| `PAST_2_WEEKS` | Last 14 days |
| `PAST_MONTH` | Last 30 days |
| `PAST_2_MONTHS` | Last 60 days |
| `PAST_YEAR` | Last 365 days |
| `PAST_2_YEARS` | Last 2 years |

**Result Types:**

| Value | Description |
|-------|-------------|
| `ONLY_LINKS` | Return only search result links |
| `LINKS_WITH_SUMMARIES` | Return links with individual summaries |
| `LINKS_WITH_FINAL_SUMMARY` | Return links with one aggregated summary |

**Response:**
```json
{
  "text": "Search results text...",
  "completion": "AI-generated summary based on search results...",
  "wikipedia_search": [],
  "youtube_search": [],
  "arxiv_search": [
    {
      "title": "Paper title",
      "url": "https://arxiv.org/abs/...",
      "summary": "Paper abstract..."
    }
  ],
  "reddit_search": [],
  "hacker_news_search": [],
  "tweets": [],
  "miner_link_scores": {}
}
```

**Example:**
```python
import os
import httpx

PROXY_URL = os.getenv("SANDBOX_PROXY_URL")
RUN_ID = os.getenv("RUN_ID")

response = httpx.post(
    f"{PROXY_URL}/api/gateway/desearch/ai/search",
    json={
        "run_id": RUN_ID,
        "prompt": "What are experts saying about AI safety?",
        "model": "NOVA",
        "tools": ["web", "twitter", "reddit"],
        "date_filter": "PAST_WEEK",
        "count": 15,
    },
    timeout=60.0,
)

result = response.json()
summary = result.get("completion", "")
tweets = result.get("tweets", [])
```

### POST /api/gateway/desearch/ai/links

Get search result links without summaries (faster than AI search).

**URL:** `{SANDBOX_PROXY_URL}/api/gateway/desearch/ai/links`

**Request Body:**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "prompt": "Climate change policy updates",
  "model": "NOVA",
  "tools": ["web", "wikipedia"],
  "count": 20
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `run_id` | string (UUID) | Yes | - | Execution tracking ID |
| `prompt` | string | Yes | - | Search query |
| `model` | string | No | `NOVA` | AI model |
| `tools` | array[string] | No | `["web"]` | Search tools (web, wikipedia, reddit, etc.) |
| `count` | integer | No | 10 | Number of links (1-100) |

**Response:**
```json
{
  "search_results": [
    {
      "title": "Result title",
      "url": "https://example.com",
      "snippet": "Preview text..."
    }
  ],
  "wikipedia_search_results": [],
  "youtube_search_results": [],
  "arxiv_search_results": [],
  "reddit_search_results": [],
  "hacker_news_search_results": []
}
```

**Example:**
```python
import httpx

response = httpx.post(
    f"{PROXY_URL}/api/gateway/desearch/ai/links",
    json={
        "run_id": RUN_ID,
        "prompt": "US inflation data 2025",
        "tools": ["web"],
        "count": 10,
    },
    timeout=30.0,
)

links = response.json().get("search_results", [])
for link in links[:5]:
    print(f"{link['title']}: {link['url']}")
```

### POST /api/gateway/desearch/web/search

Raw web search without AI processing (fastest option).

**URL:** `{SANDBOX_PROXY_URL}/api/gateway/desearch/web/search`

**Request Body:**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "query": "bitcoin price prediction",
  "num": 10,
  "start": 0
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `run_id` | string (UUID) | Yes | - | Execution tracking ID |
| `query` | string | Yes | - | Search query string |
| `num` | integer | No | 10 | Number of results (1-100) |
| `start` | integer | No | 0 | Pagination offset |

**Response:**
```json
{
  "data": [
    {
      "title": "Page title",
      "link": "https://example.com/page",
      "snippet": "Page description or excerpt...",
      "date": "2025-11-10"
    }
  ]
}
```

**Example:**
```python
import httpx

response = httpx.post(
    f"{PROXY_URL}/api/gateway/desearch/web/search",
    json={
        "run_id": RUN_ID,
        "query": "federal reserve interest rate decision",
        "num": 20,
        "start": 0,
    },
    timeout=30.0,
)

results = response.json()["data"]
for result in results:
    print(f"{result['title']}: {result['link']}")
```

### POST /api/gateway/desearch/web/crawl

Fetch and extract content from a specific URL.

**URL:** `{SANDBOX_PROXY_URL}/api/gateway/desearch/web/crawl`

**Request Body:**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "url": "https://example.com/article"
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `run_id` | string (UUID) | Yes | Execution tracking ID |
| `url` | string | Yes | Full URL to crawl |

**Response:**
```json
{
  "url": "https://example.com/article",
  "content": "Extracted text content from the page..."
}
```

**Example:**
```python
import httpx

# First, search for relevant URLs
search_response = httpx.post(
    f"{PROXY_URL}/api/gateway/desearch/web/search",
    json={"run_id": RUN_ID, "query": "climate summit outcomes", "num": 5},
    timeout=30.0,
)
urls = [r["link"] for r in search_response.json()["data"]]

# Then, crawl each URL for full content
for url in urls[:3]:
    crawl_response = httpx.post(
        f"{PROXY_URL}/api/gateway/desearch/web/crawl",
        json={"run_id": RUN_ID, "url": url},
        timeout=30.0,
    )
    content = crawl_response.json()["content"]
    # Analyze content...
```

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
    f"{PROXY_URL}/api/gateway/chutes/chat/completions",
    json={
        "run_id": "abc-123",
        "model": "deepseek-ai/DeepSeek-V3-0324",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
    },
)

# Request 2 (run_id: xyz-789, same prompt)
response2 = httpx.post(
    f"{PROXY_URL}/api/gateway/chutes/chat/completions",
    json={
        "run_id": "xyz-789",
        "model": "deepseek-ai/DeepSeek-V3-0324",
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
                f"{PROXY_URL}/api/gateway/chutes/chat/completions",
                json={
                    "run_id": RUN_ID,
                    "model": "deepseek-ai/DeepSeek-V3-0324",
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=60.0,
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]

            # Handle rate limits and cold models
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

Plan your execution time to stay within the 150-second sandbox limit:

```python
import time

start_time = time.time()
timeout_buffer = 10  # seconds
max_time = 140  # 150s limit - 10s buffer

def time_remaining():
    elapsed = time.time() - start_time
    return max_time - elapsed

# Use in your logic
if time_remaining() < 30:
    # Not enough time for API call, use fallback
    return {"event_id": event_data["event_id"], "prediction": 0.5}
```


### Model Selection

Consider using the status endpoint to select the best-performing model dynamically:

```python
def get_best_model():
    try:
        response = httpx.get(
            f"{PROXY_URL}/api/gateway/chutes/status",
            timeout=5.0,
        )

        if response.status_code == 200:
            status_list = response.json()

            # Filter for low utilization
            available = [
                s for s in status_list
                if s["utilization_current"] < 0.6 and s["rate_limit_ratio_5m"] < 0.2
            ]

            if available:
                best = min(available, key=lambda x: x["utilization_current"])
                return best["name"]
    except:
        pass

    # Fallback to reliable default
    return "deepseek-ai/DeepSeek-V3-0324"
```

### Search Strategy

Use appropriate Desearch endpoints based on your needs:

- **AI Search** (`/ai/search`): When you need summarized information
- **Links** (`/ai/links`): When you need source URLs without summaries
- **Web Search** (`/web/search`): Fastest option for raw search results
- **Crawl** (`/web/crawl`): For extracting full content from specific URLs

```python
# Multi-step search strategy
def gather_information(query: str):
    # Step 1: Fast web search for relevant URLs
    search = httpx.post(
        f"{PROXY_URL}/api/gateway/desearch/web/search",
        json={"run_id": RUN_ID, "query": query, "num": 10},
        timeout=20.0,
    )
    urls = [r["link"] for r in search.json()["data"][:5]]

    # Step 2: Crawl top results for full content
    contents = []
    for url in urls:
        crawl = httpx.post(
            f"{PROXY_URL}/api/gateway/desearch/web/crawl",
            json={"run_id": RUN_ID, "url": url},
            timeout=20.0,
        )
        if crawl.status_code == 200:
            contents.append(crawl.json()["content"][:1000])  # Truncate

    return contents
```

---

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
| `CHUTES_API_KEY not configured` | Gateway missing API key | Contact validator or check gateway configuration |
| `DESEARCH_API_KEY not configured` | Gateway missing API key | Contact validator or check gateway configuration |
| `503 Service Unavailable` | Model is cold (no active instances) | Retry with exponential backoff (2-8s delays) |
| `429 Too Many Requests` | Rate limit exceeded | Retry with exponential backoff |
| `404 Not Found` | Invalid model name | Verify model exists at https://chutes.ai/app |
| `Connection timeout` | Network issue or slow gateway | Increase timeout, implement retry logic |
| `422 Unprocessable Entity` | Invalid request parameters | Validate request body against API spec |

---

## Additional Resources

- **Chutes AI Models:** https://chutes.ai/app
- **Desearch AI Documentation:** https://desearch.ai/
- **Miner Setup Guide:** [miner-setup.md](./miner-setup.md)
- **Subnet Rules:** [subnet-rules.md](./subnet-rules.md)
- **Architecture Overview:** [architecture.md](./architecture.md)
