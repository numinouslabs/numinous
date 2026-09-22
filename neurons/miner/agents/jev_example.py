import asyncio
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
DECISIONS_URL = f"{PROXY_URL}/api/gateway/openrouter/decisions/inference"

MODEL = "typesafe/jev-1.13"
NEWS_LIMIT = 15
NEWS_MIN_IMPACT = 0.3
CORPUS_MAX_RESULTS = 5

STRENGTH_LEVELS = ["no relevant evidence", "weak", "moderate", "strong", "decisive"]
DIRECTIONS = {
    "toward_yes": "The evidence points to the event happening",
    "toward_no": "The evidence points to the event not happening",
    "mixed": "The evidence cuts both ways or is not informative",
}

TOTAL_COST = 0.0


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

    return [
        f"[{str(article.get('published_at', ''))[:10]}] "
        f"[{article.get('direction', 'neutral')}] "
        f"{article.get('headline', '')} (impact={article.get('impact_score', 0):.2f})"
        for article in articles
    ]


async def search_corpus(client: httpx.AsyncClient, event: AgentData) -> list[str]:
    global TOTAL_COST

    payload = {"run_id": RUN_ID, "query": event.title, "max_results": CORPUS_MAX_RESULTS}

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

    return [f"{item.get('title', '')} — {item.get('snippet', '')}" for item in results]


# =============================================================================
# PHASE 2: ONE JEV CALL — three typed questions about the same state
# =============================================================================
#
# Jev never writes prose, so it cannot fill the `reasoning` field an agent has to
# return. The trick is to ask for the parts of the reasoning as typed questions and
# assemble the sentence yourself — see build_reasoning() below.


def build_state(event: AgentData, news: list[str], corpus: list[str]) -> dict:
    days_left = (event.cutoff - datetime.now(event.cutoff.tzinfo)).days

    return {
        "question": event.title,
        "resolves_yes_if": event.description,
        "deadline": event.cutoff.strftime("%Y-%m-%d %H:%M UTC"),
        "days_remaining": days_left,
        "scored_news": news or ["none retrieved"],
        "background_sources": corpus or ["none retrieved"],
    }


def build_questions() -> dict:
    return {
        "will_happen": {
            "type": "noul",
            "instructions": "Will this event resolve YES by the deadline?",
            "criteria": {
                "true": "The event resolves YES by the deadline",
                "false": "The event does not resolve YES by the deadline",
            },
        },
        "direction": {
            "type": "choice",
            "instructions": "Which way does the evidence point?",
            "criteria": DIRECTIONS,
        },
        "evidence_strength": {
            "type": "score",
            "instructions": "How strong is the evidence for that direction?",
            "criteria": STRENGTH_LEVELS,
        },
    }


def build_reasoning(answers: dict) -> str:
    direction = answers["direction"]
    strength = answers["evidence_strength"]

    level_index = int(round(strength["score"]))
    level = strength["legend"].get(str(level_index), "unclear")
    direction_label = DIRECTIONS.get(direction["choice"], direction["choice"])

    summary = direction_label[0].upper() + direction_label[1:]

    return (
        f"{summary} (confidence {direction['confidence']:.2f}); "
        f"evidence is {level} at {strength['score']:.2f} on a "
        f"0-{len(STRENGTH_LEVELS) - 1} scale. "
        f"Calibrated probability {answers['will_happen']['noul']:.2f}."
    )


async def decide(client: httpx.AsyncClient, event: AgentData, state: dict) -> dict:
    global TOTAL_COST

    payload = {
        "run_id": RUN_ID,
        "model": MODEL,
        "state": state,
        "questions": build_questions(),
    }

    response = await client.post(DECISIONS_URL, json=payload)
    response.raise_for_status()
    data = response.json()

    TOTAL_COST += data.get("cost", 0.0)
    usage = data["usage"]
    print(
        f"[JEV] {usage['input_tokens']} in / {usage['output_tokens']} out "
        f"(cost=${data.get('cost', 0.0):.6f})"
    )

    answers = data["answers"]
    return {
        "event_id": event.event_id,
        "prediction": clip_probability(float(answers["will_happen"]["noul"])),
        "reasoning": build_reasoning(answers),
    }


# =============================================================================
# MAIN AGENT
# =============================================================================


async def run_agent(event: AgentData) -> dict:
    global TOTAL_COST
    TOTAL_COST = 0.0

    start_time = time.time()

    async with httpx.AsyncClient(timeout=120.0) as client:
        news, corpus = await asyncio.gather(fetch_news(client, event), search_corpus(client, event))
        state = build_state(event, news, corpus)

        try:
            result = await decide(client, event, state)
        except Exception as error:
            print(f"[JEV] Failed: {error}")
            result = {
                "event_id": event.event_id,
                "prediction": 0.5,
                "reasoning": "Jev call failed. Returning neutral prediction.",
            }

    print(f"[AGENT] Complete in {time.time() - start_time:.2f}s")
    print(f"[AGENT] Total run cost: ${TOTAL_COST:.6f}")

    return result


def agent_main(event_data: dict) -> dict:
    event = AgentData.model_validate(event_data)
    print(f"\n[AGENT] Jev typed-decision forecast for event: {event.event_id}")
    print(f"[AGENT] Title: {event.title}")

    return asyncio.run(run_agent(event))
