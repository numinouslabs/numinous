# Gateway API Reference

## Overview

The Gateway API provides miner agents with access to external services during sandbox execution. Agents run in isolated Docker containers without internet access, and the gateway acts as a controlled proxy to external APIs. Validators handle authentication, while miners can link their API accounts to cover costs and access higher budgets (see [miner-setup.md](./miner-setup.md#linking-services)).

**Track-based access:** Not all tracks have access to all endpoints. The MAIN track has full access. Other tracks are restricted to a curated subset of services — see [`track_config.py`](../neurons/validator/sandbox/signing_proxy/track_config.py) for the per-track endpoint allowlist. Requests to disallowed endpoints return 403.

**Available Services:**
- **Chutes AI**: LLM inference with multiple open-source models
- **Desearch AI**: Web search, social media search, and content crawling
- **OpenAI**: GPT-5 series models with built-in web search
- **Perplexity**: Reasoning LLMs with built-in web search
- **Vericore**: Statement verification with evidence-based metrics
- **OpenRouter**: Model router with access to hundreds of LLM models (Claude, Gemini, Llama, etc.)
- **LunarCrush**: Social media intelligence and sentiment data for any topic
- **Numinous Indicia**: Geopolitical and OSINT signals intelligence (X/Twitter, LiveUAMap)
- **Numinous Signals**: Event-relevant news signals scored by relevance and impact, causal driver graphs, and deep research reports
- **Unusual Whales**: Financial news headlines with filtering by source, ticker, and sentiment
- **Public Data Proxy**: Generic proxy any number of free public APIs across sports, economics, weather, finance, and more. No cost. Run `numi services sources` to see what's available. See [Public Data Proxy](#public-data-proxy) below.

All requests are cached to optimize performance and reduce costs.

**Cost Limits:** $0.01 (default) or $0.10 (linked account) per sandbox run for Chutes and Desearch. OpenAI: $1.00 per run (requires linked account, no free tier). Perplexity: $0.10 per run (requires linked account, no free tier). Vericore: $0.10 per run (requires linked account, no free tier). OpenRouter: $0.10 per run (requires linked account, no free tier). LunarCrush: $0.10 per run (requires linked account, no free tier). Numinous Indicia: free (no linking required). Numinous Signals: $0.10 per run (requires linked account, no free tier). Unusual Whales: $0.10 per run (requires linked account, no free tier).

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

### POST /api/gateway/desearch/x/search

Search for posts on X (Twitter) with advanced filtering options.

**URL:** `{SANDBOX_PROXY_URL}/api/gateway/desearch/x/search`

**Request Body:**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "query": "AI safety",
  "sort": "Top",
  "count": 20,
  "min_likes": 100,
  "verified": true
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `run_id` | string (UUID) | Yes | - | Execution tracking ID |
| `query` | string | Yes | - | Search query for X posts |
| `sort` | string | No | `Top` | Sort order (`Top` or `Latest`) |
| `user` | string | No | null | Filter by username |
| `start_date` | string (ISO 8601) | No | null | Filter posts after this date |
| `end_date` | string (ISO 8601) | No | null | Filter posts before this date |
| `lang` | string | No | null | Filter by language code (e.g., `en`) |
| `verified` | boolean | No | null | Filter by verified status |
| `blue_verified` | boolean | No | null | Filter by blue verified status |
| `is_quote` | boolean | No | null | Filter for quote tweets |
| `is_video` | boolean | No | null | Filter for posts with video |
| `is_image` | boolean | No | null | Filter for posts with images |
| `min_retweets` | integer | No | null | Minimum retweet count |
| `min_replies` | integer | No | null | Minimum reply count |
| `min_likes` | integer | No | null | Minimum like count |
| `count` | integer | No | 20 | Number of posts to return |

**Response:**
```json
{
  "posts": [
    {
      "id": "1234567890",
      "text": "Post content here...",
      "url": "https://x.com/user/status/1234567890",
      "created_at": "2025-01-06T12:00:00Z",
      "reply_count": 10,
      "retweet_count": 50,
      "like_count": 200,
      "view_count": 5000,
      "quote_count": 5,
      "bookmark_count": 15,
      "is_quote_tweet": false,
      "is_retweet": false,
      "lang": "en",
      "conversation_id": "1234567890",
      "media": []
    }
  ],
  "cost": 0.003
}
```

**Note:** Each post may include additional optional fields such as `in_reply_to_screen_name`, `in_reply_to_status_id`, `in_reply_to_user_id`, `quoted_status_id`, `replies`, and `display_text_range`.

**Example:**
```python
import httpx

response = httpx.post(
    f"{PROXY_URL}/api/gateway/desearch/x/search",
    json={
        "run_id": RUN_ID,
        "query": "quantum computing breakthrough",
        "sort": "Latest",
        "min_likes": 50,
        "count": 10,
    },
    timeout=30.0,
)

posts = response.json()["posts"]
for post in posts:
    print(f"{post['text'][:100]}... - {post['like_count']} likes")
```

### POST /api/gateway/desearch/x/post

Fetch detailed information about a specific X (Twitter) post by ID.

**URL:** `{SANDBOX_PROXY_URL}/api/gateway/desearch/x/post`

**Request Body:**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "post_id": "1234567890"
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `run_id` | string (UUID) | Yes | Execution tracking ID |
| `post_id` | string | Yes | The X post ID to fetch |

**Response:**
```json
{
  "user": {
    "id": "123456",
    "username": "exampleuser",
    "name": "Example User",
    "url": "https://x.com/exampleuser",
    "created_at": "2020-01-01T00:00:00Z",
    "description": "User bio...",
    "followers_count": 10000,
    "favourites_count": 5000,
    "listed_count": 100,
    "media_count": 200,
    "statuses_count": 5000,
    "verified": true,
    "is_blue_verified": false,
    "profile_image_url": "https://...",
    "profile_banner_url": "https://...",
    "location": "San Francisco, CA",
    "can_dm": true,
    "can_media_tag": true
  },
  "id": "1234567890",
  "text": "Full post content here...",
  "url": "https://x.com/exampleuser/status/1234567890",
  "created_at": "2025-01-06T12:00:00Z",
  "reply_count": 10,
  "retweet_count": 50,
  "like_count": 200,
  "view_count": 5000,
  "quote_count": 5,
  "bookmark_count": 15,
  "is_quote_tweet": false,
  "is_retweet": false,
  "lang": "en",
  "conversation_id": "1234567890",
  "media": [],
  "cost": 0.0003
}
```

**Note:** The response may include additional optional fields such as `quote` (for quote tweets), `retweet` (for retweets), `replies` (list of reply posts), `entities`, `extended_entities`, `in_reply_to_screen_name`, `in_reply_to_status_id`, `in_reply_to_user_id`, `quoted_status_id`, and `display_text_range`.

**Example:**
```python
import httpx

# Fetch a specific post
response = httpx.post(
    f"{PROXY_URL}/api/gateway/desearch/x/post",
    json={
        "run_id": RUN_ID,
        "post_id": "1234567890",
    },
    timeout=30.0,
)

post = response.json()
print(f"Author: {post['user']['username']}")
print(f"Text: {post['text']}")
print(f"Engagement: {post['like_count']} likes, {post['retweet_count']} retweets")
```

---

## OpenAI Endpoints

OpenAI provides access to GPT-5 series models with built-in web search capability.

### POST /api/gateway/openai/responses

Create a response using OpenAI's GPT-5 models with optional web search.

**URL:** `{SANDBOX_PROXY_URL}/api/gateway/openai/responses`

**Request Body:**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "model": "gpt-5-mini",
  "input": [
    {"role": "developer", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "temperature": 0.7,
  "max_output_tokens": 1000,
  "tools": [{"type": "web_search"}],
  "tool_choice": null,
  "instructions": null
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `run_id` | string (UUID) | Yes | - | Execution tracking ID from environment |
| `model` | string | Yes | - | Model identifier (see Available Models below) |
| `input` | array | Yes | - | List of message objects with `role` and `content` |
| `temperature` | float | No | 0.7 | Sampling temperature (0.0-2.0) |
| `max_output_tokens` | integer | No | null | Maximum tokens to generate |
| `tools` | array | No | null | Tool definitions (e.g., `[{"type": "web_search"}]`) |
| `tool_choice` | string/object | No | null | Tool selection strategy |
| `instructions` | string | No | null | System-level instructions |

**Available Models:**

| Model | Identifier | Notes |
|-------|-----------|-------|
| GPT-5 Mini | `gpt-5-mini` | Cost-effective, fast |
| GPT-5 | `gpt-5` | Balanced performance |
| GPT-5.2 | `gpt-5.2` | Enhanced reasoning |
| GPT-5.2 Pro | `gpt-5.2-pro` | Most capable |
| GPT-5 Nano | `gpt-5-nano` | Lightweight |

**Web Search Tool:**

Enable web search by including `tools`:
```json
"tools": [{"type": "web_search"}]
```

The model will autonomously decide when to search based on the prompt. Each search costs $0.01.

**Response:**
```json
{
  "id": "resp_123",
  "object": "response",
  "created_at": 1768496869,
  "model": "gpt-5-mini-2025-08-07",
  "output": [
    {
      "id": "msg_123",
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "The capital of France is Paris.",
          "logprobs": [],
          "annotations": []
        }
      ],
      "status": "completed"
    }
  ],
  "usage": {
    "input_tokens": 22,
    "output_tokens": 207,
    "total_tokens": 229
  },
  "status": "completed",
  "cost": 0.001953
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Response identifier |
| `model` | string | Model used for generation |
| `output` | array | List of output items (messages, reasoning steps) |
| `output[].type` | string | Item type (`message`, `reasoning`) |
| `output[].content[].text` | string | Generated text content |
| `usage` | object | Token usage statistics |
| `usage.input_tokens` | integer | Input tokens consumed |
| `usage.output_tokens` | integer | Output tokens generated |
| `cost` | float | Total cost in USD (includes token cost + web search cost) |

**Example (using httpx):**
```python
import os
import httpx

PROXY_URL = os.getenv("SANDBOX_PROXY_URL")
RUN_ID = os.getenv("RUN_ID")

response = httpx.post(
    f"{PROXY_URL}/api/gateway/openai/responses",
    json={
        "run_id": RUN_ID,
        "model": "gpt-5-mini",
        "input": [
            {"role": "developer", "content": "You are an expert forecaster."},
            {"role": "user", "content": "What is the probability of rain tomorrow?"}
        ],
        "tools": [{"type": "web_search"}],
        "temperature": 0.7,
    },
    timeout=120.0,
)

result = response.json()

# Extract text from output
for item in result["output"]:
    if item["type"] == "message":
        for content in item["content"]:
            if content.get("text"):
                print(content["text"])
```

**Cost Calculation:**

Total cost = Token cost + Web search cost

- **Token cost:** Based on input/output tokens and model pricing
- **Web search cost:** $0.01 per search executed

The `cost` field in the response includes both components.

**Error Handling:**

| Status Code | Description | Recommended Action |
|-------------|-------------|-------------------|
| 503 | Service Unavailable | Retry with exponential backoff |
| 404 | Model not found | Verify model identifier |
| 429 | Rate limit exceeded | Retry with exponential backoff |
| 401 | Authentication failed | Contact validator |
| 500 | Internal server error | Retry with fallback |

**Best Practices:**

1. **Use web_search selectively:** Only enable when research is needed
2. **Clear prompts:** Explicitly ask the model to search before forecasting
3. **Model selection:** Use `gpt-5-mini` for cost-efficiency, `gpt-5.2` for complex reasoning
4. **Error handling:** Always implement retry logic with fallback predictions

---

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

**Difference from `/responses`:**

| Feature | `/responses` | `/responses/inference` |
|---------|-------------|----------------------|
| `web_search` built-in | Allowed | Blocked |
| Custom `function` tools | Allowed | Allowed |
| Other built-in tools | Blocked | Blocked |

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

> **Note:** Miners link the same OpenAI API key for both endpoints via `numi services link openai`. The key is stored once and reused across `/responses` and `/responses/inference`.

---

## Perplexity Endpoints

Perplexity provides reasoning LLMs with built-in web search capability.

### POST /api/gateway/perplexity/chat/completions

Create a response using Perplexity's reasoning models with automatic web search.

**URL:** `{SANDBOX_PROXY_URL}/api/gateway/perplexity/chat/completions`

**Request Body:**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "model": "sonar-reasoning-pro",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "temperature": 0.7,
  "max_tokens": 1000,
  "search_recency_filter": "month"
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
| `search_recency_filter` | string | No | null | Time range for search results (`day`, `week`, `month`, `year`) |

**Available Models:**

| Model | Identifier | Notes |
|-------|-----------|-------|
| Sonar Reasoning Pro | `sonar-reasoning-pro` | Most capable reasoning model |
| Sonar Pro | `sonar-pro` | Balanced performance |
| Sonar | `sonar` | Fast and efficient |

**Response:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "sonar-reasoning-pro",
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
  },
  "citations": [
    "https://example.com/source1",
    "https://example.com/source2"
  ],
  "search_results": [
    {
      "title": "Source title",
      "url": "https://example.com/source1",
      "snippet": "Relevant text..."
    }
  ],
  "cost": 0.002145
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Response identifier |
| `model` | string | Model used for generation |
| `choices` | array | List of completion choices |
| `choices[].message.content` | string | Generated text content |
| `usage` | object | Token usage statistics |
| `citations` | array | List of source URLs used |
| `search_results` | array | Detailed search result objects |
| `cost` | float | Total cost in USD |

**Example (using httpx):**
```python
import os
import httpx

PROXY_URL = os.getenv("SANDBOX_PROXY_URL")
RUN_ID = os.getenv("RUN_ID")

response = httpx.post(
    f"{PROXY_URL}/api/gateway/perplexity/chat/completions",
    json={
        "run_id": RUN_ID,
        "model": "sonar-reasoning-pro",
        "messages": [
            {"role": "system", "content": "You are an expert forecaster."},
            {"role": "user", "content": "What is the probability of rain tomorrow?"}
        ],
        "temperature": 0.2,
        "search_recency_filter": "day",
    },
    timeout=120.0,
)

result = response.json()

content = result["choices"][0]["message"]["content"]
citations = result.get("citations", [])

print(f"Response: {content}")
print(f"Sources: {citations}")
```

**Error Handling:**

| Status Code | Description | Recommended Action |
|-------------|-------------|-------------------|
| 503 | Service Unavailable | Retry with exponential backoff |
| 404 | Model not found | Verify model identifier |
| 429 | Rate limit exceeded | Retry with exponential backoff |
| 401 | Authentication failed | Contact validator |
| 500 | Internal server error | Retry with fallback |

**Best Practices:**

1. **Use search_recency_filter:** Set to `day` or `week` for time-sensitive events
2. **Extract citations:** Use the `citations` array to verify information sources
3. **Model selection:** Use `sonar-reasoning-pro` for complex reasoning tasks
4. **Error handling:** Always implement retry logic with fallback predictions

**Note:** Perplexity has no free tier. You must link your API key to use Perplexity models.

---

## Vericore Endpoints

Vericore provides statement verification with evidence-based metrics including sentiment, conviction, source credibility, and more.

### POST /api/gateway/vericore/calculate-rating

Verify a statement against web evidence and get detailed metrics.

**URL:** `{SANDBOX_PROXY_URL}/api/gateway/vericore/calculate-rating`

**Request Body:**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "statement": "Bitcoin will reach $100k by end of 2026",
  "generate_preview": false
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `run_id` | string (UUID) | Yes | - | Execution tracking ID from environment |
| `statement` | string | Yes | - | Statement to verify against web evidence |
| `generate_preview` | boolean | No | false | Generate a preview URL for the results |

**Response:**
```json
{
  "batch_id": "mlzjxglo15m23k",
  "request_id": "req-mlzjxgmc4amr6",
  "preview_url": "",
  "evidence_summary": {
    "total_count": 12,
    "neutral": 37.5,
    "entailment": 1.03,
    "contradiction": 61.46,
    "sentiment": -0.07,
    "conviction": 0.82,
    "source_credibility": 0.93,
    "narrative_momentum": 0.48,
    "risk_reward_sentiment": -0.15,
    "political_leaning": 0.0,
    "catalyst_detection": 0.12,
    "statements": [
      {
        "statement": "Evidence text from source...",
        "url": "https://example.com/article",
        "contradiction": 0.87,
        "neutral": 0.12,
        "entailment": 0.01,
        "sentiment": -0.5,
        "conviction": 0.75,
        "source_credibility": 0.85,
        "narrative_momentum": 0.5,
        "risk_reward_sentiment": -0.5,
        "political_leaning": 0.0,
        "catalyst_detection": 0.3
      }
    ]
  },
  "cost": 0.05
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `batch_id` | string | Batch identifier |
| `request_id` | string | Request identifier |
| `preview_url` | string | Preview URL (empty if `generate_preview` is false) |
| `evidence_summary.total_count` | integer | Number of evidence sources found |
| `evidence_summary.entailment` | float | Aggregated entailment score |
| `evidence_summary.contradiction` | float | Aggregated contradiction score |
| `evidence_summary.sentiment` | float | Aggregated sentiment (-1.0 to 1.0) |
| `evidence_summary.conviction` | float | Aggregated conviction level |
| `evidence_summary.source_credibility` | float | Average source credibility |
| `evidence_summary.statements` | array | Individual evidence sources with per-source metrics |

**Example (using httpx):**
```python
import os
import httpx

PROXY_URL = os.getenv("SANDBOX_PROXY_URL")
RUN_ID = os.getenv("RUN_ID")

response = httpx.post(
    f"{PROXY_URL}/api/gateway/vericore/calculate-rating",
    json={
        "run_id": RUN_ID,
        "statement": "Bitcoin will reach $100k by end of 2026",
    },
    timeout=120.0,
)

result = response.json()

summary = result["evidence_summary"]
total = summary["total_count"]
contradiction = summary["contradiction"]
sentiment = summary["sentiment"]
conviction = summary["conviction"]
credibility = summary["source_credibility"]
```

**Error Handling:**

| Status Code | Description | Recommended Action |
|-------------|-------------|-------------------|
| 503 | Service Unavailable | Retry with exponential backoff |
| 429 | Rate limit exceeded | Retry with exponential backoff |
| 401 | Authentication failed | Contact validator |
| 500 | Internal server error | Retry with fallback |

**Note:** Vericore has no free tier. You must link your API key to use Vericore. Each call costs $0.05.

---

## OpenRouter Endpoints

OpenRouter is a model router that provides access to hundreds of LLM models through a unified API. You can use models from Anthropic, Google, Meta, and many other providers.

### POST /api/gateway/openrouter/chat/completions

Generate chat completions using any OpenRouter-supported model.

**URL:** `{SANDBOX_PROXY_URL}/api/gateway/openrouter/chat/completions`

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
| `model` | string | Yes | - | OpenRouter model ID (e.g., `anthropic/claude-sonnet-4-6`) |
| `messages` | array | Yes | - | Chat messages array with `role` and `content` |
| `temperature` | float | No | 0.7 | Sampling temperature (0.0-2.0) |
| `max_tokens` | integer | No | - | Maximum tokens to generate |
| `tools` | array | No | - | Tool/function definitions for function calling |
| `tool_choice` | string/object | No | - | Tool selection mode |

**Response:**
```json
{
  "id": "gen-abc123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "anthropic/claude-sonnet-4-6",
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
    "prompt_tokens": 25,
    "completion_tokens": 10,
    "total_tokens": 35,
    "cost": 0.000135
  },
  "cost": 0.000135
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique completion identifier |
| `model` | string | Model used for the completion |
| `choices` | array | Array of completion choices |
| `choices[].message.content` | string | Generated text response |
| `choices[].finish_reason` | string | Why generation stopped (`stop`, `length`, `tool_calls`) |
| `usage.prompt_tokens` | integer | Input tokens used |
| `usage.completion_tokens` | integer | Output tokens generated |
| `usage.cost` | decimal | Actual cost reported by OpenRouter |
| `cost` | decimal | Total cost for this request |

**Popular Models:**

| Model ID | Description |
|----------|-------------|
| `anthropic/claude-sonnet-4-6` | Claude Sonnet 4.6 - balanced performance |
| `anthropic/claude-haiku-4-5` | Claude Haiku 4.5 - fast and cost-effective |
| `google/gemini-2.5-flash` | Gemini 2.5 Flash - fast and affordable |
| `google/gemini-2.5-pro` | Gemini 2.5 Pro - high capability |

See the full model list at https://openrouter.ai/models

**Example (using httpx):**
```python
import os
import httpx

PROXY_URL = os.getenv("SANDBOX_PROXY_URL")
RUN_ID = os.getenv("RUN_ID")

response = httpx.post(
    f"{PROXY_URL}/api/gateway/openrouter/chat/completions",
    json={
        "run_id": RUN_ID,
        "model": "anthropic/claude-sonnet-4-6",
        "messages": [
            {"role": "user", "content": "Analyze the likelihood of this event..."}
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
    },
    timeout=120.0,
)

result = response.json()
content = result["choices"][0]["message"]["content"]
cost = result.get("cost", 0.0)
```

**Error Handling:**

| Status Code | Description | Recommended Action |
|-------------|-------------|-------------------|
| 503 | Service Unavailable | Retry with exponential backoff |
| 429 | Rate limit exceeded | Retry with exponential backoff |
| 401 | Authentication failed | Contact validator |
| 500 | Internal server error | Retry with fallback model |

**Note:** OpenRouter has no free tier. You must link your API key to use OpenRouter models.

---

## LunarCrush Endpoints

LunarCrush provides social media intelligence and sentiment data for any topic -- crypto, geopolitics, stocks, and more. Useful for gauging public sentiment and social trends around events.

### POST /api/gateway/lunar-crush/whatsup

AI-generated sentiment summary with bullish/bearish themes and percentages.

**URL:** `{SANDBOX_PROXY_URL}/api/gateway/lunar-crush/whatsup`

**Request Body:**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "topic": "bitcoin"
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `run_id` | string (UUID) | Yes | - | Execution tracking ID from environment |
| `topic` | string | Yes | - | Topic to search (lowercase, e.g., "bitcoin", "iran", "oil prices") |

**Response:**
```json
{
  "config": {"id": "bitcoin", "name": "Bitcoin", "symbol": "BTC"},
  "summary": "Bitcoin surges past $75K as institutional buying drives demand.",
  "supportive": [
    {"title": "Institutional Adoption", "description": "Major companies increasing holdings", "percent": 25.0}
  ],
  "critical": [
    {"title": "Regulatory Concerns", "description": "Ongoing debates about regulation", "percent": 5.0}
  ],
  "cost": 0.001
}
```

### POST /api/gateway/lunar-crush/topic

Current social metrics, rank, and trend for a topic.

**URL:** `{SANDBOX_PROXY_URL}/api/gateway/lunar-crush/topic`

**Request Body:**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "topic": "bitcoin"
}
```

**Response:**
```json
{
  "config": {"id": "bitcoin"},
  "data": {
    "topic": "bitcoin",
    "title": "Bitcoin",
    "topic_rank": 83,
    "interactions_24h": 367838294,
    "num_contributors": 71451,
    "num_posts": 292738,
    "trend": "flat",
    "categories": ["Cryptocurrencies"]
  },
  "cost": 0.001
}
```

### POST /api/gateway/lunar-crush/news

Top news articles with sentiment scores for a topic.

**URL:** `{SANDBOX_PROXY_URL}/api/gateway/lunar-crush/news`

**Request Body:**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "topic": "bitcoin"
}
```

**Response (truncated):**
```json
{
  "config": {"id": "bitcoin"},
  "data": [
    {
      "id": "news-123",
      "post_type": "news",
      "post_title": "Crypto ETFs Rally Continues With $202 Million Inflow",
      "post_sentiment": 3.22,
      "creator_name": "BitcoinNews",
      "creator_followers": 3273394,
      "interactions_24h": 2790
    }
  ],
  "cost": 0.001
}
```

### POST /api/gateway/lunar-crush/time-series

Historical daily or hourly social + price data.

**URL:** `{SANDBOX_PROXY_URL}/api/gateway/lunar-crush/time-series`

**Request Body:**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "topic": "bitcoin",
  "bucket": "day"
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `topic` | string | Yes | - | Topic to search |
| `bucket` | string | No | "day" | Aggregation: "day" or "hour" |

**Response (truncated):**
```json
{
  "config": {"bucket": "day", "topic": "bitcoin"},
  "data": [
    {
      "time": 1773662400,
      "close": 75000.0,
      "sentiment": 80,
      "galaxy_score": 65.0,
      "interactions": 500000,
      "volume_24h": 53800000000
    }
  ],
  "cost": 0.001
}
```

### POST /api/gateway/lunar-crush/posts

Top social media posts across platforms (Twitter, YouTube, TikTok, Reddit, Instagram).

**URL:** `{SANDBOX_PROXY_URL}/api/gateway/lunar-crush/posts`

**Request Body:**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "topic": "bitcoin",
  "start": null,
  "end": null
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `topic` | string | Yes | - | Topic to search |
| `start` | integer | No | null | Unix timestamp filter start |
| `end` | integer | No | null | Unix timestamp filter end |

### POST /api/gateway/lunar-crush/coins-list

Market + social data for all 6600+ tracked cryptocurrencies.

**URL:** `{SANDBOX_PROXY_URL}/api/gateway/lunar-crush/coins-list`

**Request Body:**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Response (truncated):**
```json
{
  "config": {"sort": "market_cap_rank", "total_rows": 6668},
  "data": [
    {
      "id": 1,
      "symbol": "BTC",
      "name": "Bitcoin",
      "price": 74400.08,
      "market_cap_rank": 1,
      "sentiment": 81,
      "galaxy_score": 61.8,
      "social_volume_24h": 292342,
      "percent_change_24h": 0.49
    }
  ],
  "cost": 0.001
}
```

**Example (using httpx):**
```python
import os
import httpx

PROXY_URL = os.getenv("SANDBOX_PROXY_URL")
RUN_ID = os.getenv("RUN_ID")

# Get AI sentiment summary
whatsup = httpx.post(
    f"{PROXY_URL}/api/gateway/lunar-crush/whatsup",
    json={"run_id": RUN_ID, "topic": "bitcoin"},
    timeout=30.0,
).json()

summary = whatsup["summary"]
bullish_pct = sum(t["percent"] for t in whatsup["supportive"])
bearish_pct = sum(t["percent"] for t in whatsup["critical"])

# Get current social stats
topic = httpx.post(
    f"{PROXY_URL}/api/gateway/lunar-crush/topic",
    json={"run_id": RUN_ID, "topic": "bitcoin"},
    timeout=30.0,
).json()

trend = topic["data"]["trend"]  # "up", "flat", or "down"
```

**Error Handling:**

| Status Code | Description | Recommended Action |
|-------------|-------------|-------------------|
| 404 | Topic not found | Try a simpler/shorter topic keyword |
| 429 | Budget limit exceeded | Link your API key |
| 503 | Service unavailable | Retry with backoff |

**Note:** LunarCrush has no free tier. You must link your API key to use LunarCrush. Subscription-based pricing, nominal $0.001 per call for budget tracking.

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

Numinous Signals scores recent news and events by relevance and impact to a given question or market, and returns ranked signals. Useful for getting structured context before making a prediction.

### POST /api/gateway/numinous-signals/signals

Compute scored signals for a question or Polymarket market.

**URL:** `{SANDBOX_PROXY_URL}/api/gateway/numinous-signals/signals`

**Request Body:**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "question": "Will Bitcoin exceed 100k by end of March 2026?",
  "relevance_threshold": 0.25,
  "max_events_per_source": 10,
  "time_window_hours": 72
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `run_id` | string (UUID) | Yes | - | Execution tracking ID from environment |
| `market` | string | One of market/question | - | Polymarket URL, slug, or condition ID |
| `question` | string | One of market/question | - | Free-text question to find signals for |
| `relevance_threshold` | float | No | 0.25 | Minimum relevance score to include (0.0-1.0) |
| `max_events_per_source` | integer | No | 25 | Cap per source before dedup (1-100) |
| `time_window_hours` | integer | No | 72 | Lookback window in hours (1-720) |

Provide exactly one of `market` or `question`, not both.

**Response:**
```json
{
  "signals": [
    {
      "headline": "Bitcoin Price Prediction March 2026",
      "source": "perplexity",
      "timestamp": "2026-03-27T19:50:27.802990Z",
      "relevance_score": 1.0,
      "impact_score": 0.5,
      "direction": "neutral",
      "rationale": "Directly addresses BTC price prediction for March 2026.",
      "source_url": "https://example.com/btc-prediction"
    }
  ],
  "source_weights": [
    {
      "source_name": "perplexity",
      "event_count": 5,
      "avg_relevance_score": 0.68,
      "avg_impact_score": 0.36
    }
  ],
  "total_event_count": 20,
  "filtered_count": 16,
  "failed_sources": ["exa"],
  "question_context": "Will Bitcoin exceed 100k by end of March 2026?",
  "processing_metadata": {
    "duration_seconds": 24.084,
    "llm_scored_count": 20,
    "total_ingested_events": 20,
    "question_text": "Will Bitcoin exceed 100k by end of March 2026?",
    "market_yes_price": null
  },
  "cost": 0.001
}
```

**Signal Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `signals` | array | Scored signals sorted by relevance |
| `signals[].headline` | string | Signal headline |
| `signals[].source` | string | Source that produced this signal |
| `signals[].timestamp` | string (ISO 8601) | When the signal was captured |
| `signals[].relevance_score` | float | How relevant to the question (0.0-1.0) |
| `signals[].impact_score` | float | How much it moves the needle (0.0-1.0) |
| `signals[].direction` | string | `supports_yes`, `supports_no`, or `neutral` |
| `signals[].rationale` | string | One-sentence explanation of the score |
| `signals[].source_url` | string | Original source URL (may be null) |
| `source_weights` | array | Per-source aggregated stats |
| `total_event_count` | integer | Events after dedup, before threshold filtering |
| `filtered_count` | integer | Events removed by relevance threshold |
| `failed_sources` | array | Sources that failed to respond |
| `cost` | float | Cost for this request ($0.001) |

**Example (using httpx):**
```python
import os
import httpx

PROXY_URL = os.getenv("SANDBOX_PROXY_URL")
RUN_ID = os.getenv("RUN_ID")

SIGNALS_URL = f"{PROXY_URL}/api/gateway/numinous-signals"

response = httpx.post(
    f"{SIGNALS_URL}/signals",
    json={
        "run_id": RUN_ID,
        "question": "Will Bitcoin exceed 100k by end of March 2026?",
        "max_events_per_source": 10,
        "time_window_hours": 48,
    },
    timeout=120.0,
)

data = response.json()
for s in data["signals"]:
    print(f"[{s['direction']}] {s['headline']} (relevance={s['relevance_score']:.2f})")
```

**Error Handling:**

| Status Code | Description | Recommended Action |
|-------------|-------------|-------------------|
| 422 | Validation error (e.g., both market and question provided) | Fix request parameters |
| 429 | Budget limit exceeded | Reduce calls per run |
| 503 | Service Unavailable | Retry with exponential backoff |
| 500 | Internal server error | Retry with fallback |

**Note:** Numinous Signals requires linking your Eversight API key. There is no free tier — you must link your account to use this endpoint. Get your API key at [eversight.numinouslabs.io/api-keys](https://eversight.numinouslabs.io/api-keys). This endpoint can take up to 30 seconds to respond. Set your timeout accordingly (120s recommended).

See `neurons/miner/agents/signals_openai_example.py` for a complete agent that combines Numinous Signals with OpenAI web search for forecasting.

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

---

## Unusual Whales Endpoints

Unusual Whales provides financial news headlines with filtering by source, ticker, and sentiment. Useful for tracking market-moving news and earnings-related events.

### POST /api/gateway/unusual-whales/news/headlines

Fetch financial news headlines with optional filtering.

**URL:** `{SANDBOX_PROXY_URL}/api/gateway/unusual-whales/news/headlines`

**Request Body:**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "ticker": "AAPL",
  "major_only": true,
  "limit": 20,
  "page": 0
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `run_id` | string (UUID) | Yes | - | Execution tracking ID from environment |
| `sources` | string | No | null | Comma-separated news sources to filter by |
| `search_term` | string | No | null | Search term to filter headlines |
| `ticker` | string | No | null | Ticker symbol to filter headlines (e.g. `AAPL`) |
| `major_only` | boolean | No | null | Only return major headlines |
| `limit` | integer | No | 50 | Number of headlines per page (1-200) |
| `page` | integer | No | 0 | Page number for pagination |

**Response:**
```json
{
  "headlines": [
    {
      "headline": "Apple Reports Record Q1 Earnings",
      "source": "benzinga",
      "created_at": "2026-04-08T14:30:00Z",
      "is_major": true,
      "sentiment": "positive",
      "tickers": ["AAPL"],
      "tags": ["earnings", "tech"],
      "meta": {}
    }
  ],
  "cost": 0.0001
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `headlines` | array | List of news headline objects |
| `headlines[].headline` | string | Headline text |
| `headlines[].source` | string | News source (e.g. `benzinga`, `reuters`) |
| `headlines[].created_at` | string (ISO 8601) | When the headline was published |
| `headlines[].is_major` | boolean | Whether this is a major headline |
| `headlines[].sentiment` | string | Sentiment (`positive`, `negative`, `neutral`, or null) |
| `headlines[].tickers` | array | Related ticker symbols |
| `headlines[].tags` | array | Tags/categories |
| `cost` | float | Cost for this request ($0.0001) |

**Example (using httpx):**
```python
import os
import httpx

PROXY_URL = os.getenv("SANDBOX_PROXY_URL")
RUN_ID = os.getenv("RUN_ID")

response = httpx.post(
    f"{PROXY_URL}/api/gateway/unusual-whales/news/headlines",
    json={
        "run_id": RUN_ID,
        "ticker": "AAPL",
        "major_only": True,
        "limit": 20,
    },
    timeout=60.0,
)

result = response.json()
for headline in result["headlines"]:
    print(f"[{headline['sentiment']}] {headline['headline']} ({headline['source']})")
```

**Error Handling:**

| Status Code | Description | Recommended Action |
|-------------|-------------|-------------------|
| 401 | API key not configured | Link your Unusual Whales API key |
| 429 | Budget limit exceeded | Reduce calls per run |
| 503 | Service Unavailable | Retry with exponential backoff |
| 500 | Internal server error | Retry with fallback |

**Note:** Unusual Whales requires linking your API key. There is no free tier — you must link your account to use this endpoint. Get your API key at [unusualwhales.com/pricing](https://unusualwhales.com/pricing?product=api).

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

## Public Data Proxy

The public data proxy gives agents access to a curated list of public APIs through a single generic endpoint. No API key linking required for most sources. Some sources require you to link your own API key — run `numi services sources` to see which.

**Available on:** MAIN and SIGNAL tracks.

**Cost:** Always $0.00.

### Discovering Available Sources

Run this from your terminal to see what sources are currently available:

```bash
numi services sources
```

This shows all sources grouped by whether they require an API key. The list is updated over time as new sources are added — always check this rather than hardcoding assumptions.

For sources that require an API key, link yours with:

```bash
numi services link <source-name>
```

### POST /api/gateway/public-data/fetch

Proxy a request to any whitelisted public data source.

**URL:** `{SANDBOX_PROXY_URL}/api/gateway/public-data/fetch`

**Request Body:**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "url": "https://your-whitelisted-source.com/api/endpoint",
  "method": "GET",
  "headers": {},
  "query_params": {},
  "body": null,
  "timeout": 30.0
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `run_id` | UUID | Yes | Current run identifier |
| `url` | string | Yes | Full URL of the public API to call |
| `method` | string | No | HTTP method: `GET`, `POST`, `PUT`, `DELETE`. Default: `GET` |
| `headers` | object | No | Additional request headers |
| `query_params` | object | No | Query string parameters |
| `body` | string | No | Request body (raw string) |
| `timeout` | float | No | Timeout in seconds (1–60). Default: 30 |

**Response:**
```json
{
  "status_code": 200,
  "response_headers": {"content-type": "application/json"},
  "response_body": "{...}",
  "content_type": "application/json",
  "source_name": "source_name",
  "source_category": "category",
  "cost": 0.0
}
```

The `response_body` is the raw response text (up to 5MB). Parse it as needed.

**Error Responses:**
- `400 Bad Request` — URL domain not in whitelist, or URL resolves to a private IP
- `503 Service Unavailable` — upstream API unreachable

**Example:**
```python
import os
import httpx

PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
RUN_ID = os.getenv("RUN_ID")

def fetch_nfl_scoreboard() -> dict:
    response = httpx.post(
        f"{PROXY_URL}/api/gateway/public-data/fetch",
        json={
            "run_id": RUN_ID,
            "url": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
        },
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["response_body"]
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
- **LunarCrush API:** https://lunarcrush.com/developers/api/endpoints
- **Unusual Whales API:** https://unusualwhales.com/public-api
- **Miner Setup Guide:** [miner-setup.md](./miner-setup.md)
- **Subnet Rules:** [subnet-rules.md](./subnet-rules.md)
- **Architecture Overview:** [architecture.md](./architecture.md)
