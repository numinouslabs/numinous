import asyncio
import json
import os
import time
from datetime import datetime

import httpx
from pydantic import BaseModel

RUN_ID = os.getenv("RUN_ID")
if not RUN_ID:
    raise ValueError("RUN_ID environment variable is required but not set")

PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
SIGNALS_URL = f"{PROXY_URL}/api/gateway/numinous-signals"

NEWS_LIMIT = 15
NEWS_MIN_IMPACT = 0.3
CORPUS_MAX_RESULTS = 5

TOTAL_COST = 0.0

FORECAST_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "forecast",
        "schema": {
            "type": "object",
            "properties": {
                "probability": {"type": "number"},
                "reasoning": {"type": "string"},
            },
            "required": ["probability", "reasoning"],
            "additionalProperties": False,
        },
    },
}


class AgentData(BaseModel):
    event_id: str
    title: str
    description: str
    cutoff: datetime
    metadata: dict


def clip_probability(prediction: float) -> float:
    return max(0.0, min(1.0, prediction))


# =============================================================================
# PHASE 1: EVIDENCE PACK — scored news + corpus search
# =============================================================================


async def fetch_news(client: httpx.AsyncClient, event: AgentData) -> list[str]:
    global TOTAL_COST

    payload = {
        "run_id": RUN_ID,
        "event_id": event.event_id,
        "min_impact_score": NEWS_MIN_IMPACT,
        "order": "impact",
        "limit": NEWS_LIMIT,
    }

    try:
        response = await client.post(f"{SIGNALS_URL}/news", json=payload)
        response.raise_for_status()
        data = response.json()
    except Exception as error:
        print(f"[NEWS] Failed: {error}")
        return []

    TOTAL_COST += data.get("cost", 0.0)
    articles = data.get("articles", [])
    print(f"[NEWS] {len(articles)} scored articles (cost=${data.get('cost', 0.0):.6f})")

    lines = []
    for article in articles:
        published = str(article.get("published_at", ""))[:10]
        lines.append(
            f"- [{published}] [{article.get('direction', 'neutral')}] "
            f"{article.get('headline', '')} "
            f"(impact={article.get('impact_score', 0):.2f}) "
            f"— {article.get('rationale', '')}"
        )
    return lines


async def search_corpus(client: httpx.AsyncClient, event: AgentData) -> list[str]:
    global TOTAL_COST

    payload = {
        "run_id": RUN_ID,
        "query": event.title,
        "max_results": CORPUS_MAX_RESULTS,
    }

    try:
        response = await client.post(f"{SIGNALS_URL}/corpus/search", json=payload)
        response.raise_for_status()
        data = response.json()
    except Exception as error:
        print(f"[CORPUS] Failed: {error}")
        return []

    TOTAL_COST += data.get("cost", 0.0)
    results = data.get("results", [])
    print(f"[CORPUS] {len(results)} sources (cost=${data.get('cost', 0.0):.6f})")

    return [f"- {item.get('title', '')} — {item.get('snippet', '')}" for item in results]


def build_evidence_pack(news_lines: list[str], corpus_lines: list[str]) -> str:
    sections = []
    if news_lines:
        sections.append("SCORED NEWS:\n" + "\n".join(news_lines))
    if corpus_lines:
        sections.append("BACKGROUND SOURCES:\n" + "\n".join(corpus_lines))
    if not sections:
        return "No evidence retrieved."
    return "\n\n".join(sections)


# =============================================================================
# PHASE 2: NUMINOUS-1 STRUCTURED FORECAST
# =============================================================================


def build_prompt(event: AgentData, evidence: str) -> str:
    return f"""QUESTION: {event.title}

RESOLVES YES IF: {event.description}

DEADLINE: {event.cutoff.strftime('%Y-%m-%d %H:%M UTC')}
TODAY: {datetime.utcnow().strftime('%Y-%m-%d')}

EVIDENCE:
{evidence}

Weigh the evidence against the time remaining and commit to a calibrated probability."""


async def forecast(client: httpx.AsyncClient, event: AgentData, evidence: str) -> dict:
    global TOTAL_COST

    payload = {
        "run_id": RUN_ID,
        "messages": [{"role": "user", "content": build_prompt(event, evidence)}],
        "max_tokens": 400,
        "temperature": 0.0,
        "response_format": FORECAST_SCHEMA,
    }

    response = await client.post(f"{SIGNALS_URL}/numinous-1/chat/completions", json=payload)
    response.raise_for_status()
    data = response.json()

    TOTAL_COST += data.get("cost", 0.0)
    print(
        f"[NUMINOUS-1] {data['usage']['prompt_tokens']} in / "
        f"{data['usage']['completion_tokens']} out (cost=${data.get('cost', 0.0):.6f})"
    )

    parsed = json.loads(data["choices"][0]["message"]["content"])
    return {
        "event_id": event.event_id,
        "prediction": clip_probability(float(parsed["probability"])),
        "reasoning": parsed.get("reasoning", ""),
    }


# =============================================================================
# MAIN AGENT
# =============================================================================


async def run_agent(event: AgentData) -> dict:
    global TOTAL_COST
    TOTAL_COST = 0.0

    start_time = time.time()

    async with httpx.AsyncClient(timeout=120.0) as client:
        news_lines, corpus_lines = await asyncio.gather(
            fetch_news(client, event), search_corpus(client, event)
        )
        evidence = build_evidence_pack(news_lines, corpus_lines)

        try:
            result = await forecast(client, event, evidence)
        except Exception as error:
            print(f"[NUMINOUS-1] Failed: {error}")
            result = {
                "event_id": event.event_id,
                "prediction": 0.5,
                "reasoning": "Numinous-1 call failed. Returning neutral prediction.",
            }

    print(f"[AGENT] Complete in {time.time() - start_time:.2f}s")
    print(f"[AGENT] Total run cost: ${TOTAL_COST:.6f}")

    return result


def agent_main(event_data: dict) -> dict:
    event = AgentData.model_validate(event_data)
    print(f"\n[AGENT] Numinous-1 evidence-pack forecast for event: {event.event_id}")
    print(f"[AGENT] Title: {event.title}")

    return asyncio.run(run_agent(event))
