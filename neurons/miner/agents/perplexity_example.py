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
PERPLEXITY_URL = f"{PROXY_URL}/api/gateway/perplexity"

PERPLEXITY_MODELS = [
    "sonar-reasoning-pro",
    "sonar-pro",
    "sonar",
]

MAX_RETRIES = 3
BASE_BACKOFF = 1.5

TOTAL_COST = 0.0


class AgentData(BaseModel):
    event_id: str
    title: str
    description: str
    cutoff: datetime
    metadata: dict


async def retry_with_backoff(func, max_retries: int = MAX_RETRIES):
    for attempt in range(max_retries):
        try:
            return await func()
        except httpx.TimeoutException as e:
            if attempt < max_retries - 1:
                delay = BASE_BACKOFF ** (attempt + 1)
                print(f"[RETRY] Timeout, retrying in {delay}s...")
                await asyncio.sleep(delay)
            else:
                raise Exception(f"Max retries exceeded: {e}")
        except httpx.HTTPStatusError as e:
            try:
                error_detail = e.response.json().get("detail", str(e))
            except Exception:
                error_detail = e.response.text if hasattr(e.response, "text") else str(e)

            if e.response.status_code == 429:
                if attempt < max_retries - 1:
                    delay = BASE_BACKOFF ** (attempt + 1)
                    print(f"[RETRY] Rate limited (429), retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    raise Exception(f"Rate limit exceeded: {error_detail}")
            else:
                raise Exception(f"HTTP {e.response.status_code}: {error_detail}")
        except Exception:
            raise


def clip_probability(prediction: float) -> float:
    return max(0.0, min(1.0, prediction))


def build_forecast_messages(event: AgentData) -> list[dict]:
    cutoff_date = event.cutoff.strftime("%Y-%m-%d %H:%M UTC")

    system_prompt = """You are an expert forecaster specializing in probabilistic predictions.
Your task is to estimate the likelihood of binary events (YES/NO outcomes).

You will receive search results with current information about the event.

Key principles:
- Consider base rates and historical precedents
- Weigh evidence quality and recency
- Account for uncertainty and missing information
- Avoid extreme predictions (0 or 1) unless evidence is overwhelming
- Use the full probability range: 0.0 (impossible) to 1.0 (certain)"""

    user_prompt = f"""**Event to Forecast:**
{event.title}

**Full Description:**
{event.description}

**Forecast Deadline:** {cutoff_date}

**Your Task:**
Based on recent news, data, and expert opinions, estimate the probability (0.0 to 1.0) that this event will occur or resolve as YES by the deadline.

Search for and consider:
1. Recent news and developments related to this event
2. Expert analysis, predictions, or market sentiment
3. Historical data or precedents for similar events
4. Current trends and momentum

Then provide:
- What is the base rate for similar events?
- What specific evidence supports or contradicts this outcome?
- What uncertainties or unknowns remain?
- How confident are you in available information?

**Required Output Format:**
PREDICTION: [number between 0.0 and 1.0]
REASONING: [2-4 sentences explaining your probability estimate, key factors considered, and main uncertainties]"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


async def forecast_with_perplexity(event: AgentData) -> dict:
    global TOTAL_COST
    print("[FORECAST] Generating forecast with Perplexity...")

    messages = build_forecast_messages(event)

    for i, model in enumerate(PERPLEXITY_MODELS):
        print(f"[FORECAST] Trying model {i+1}/{len(PERPLEXITY_MODELS)}: {model}")

        try:

            async def perplexity_call():
                async with httpx.AsyncClient(timeout=120.0) as client:
                    payload = {
                        "model": model,
                        "messages": messages,
                        "temperature": 0.2,
                        "search_recency_filter": "month",
                        "run_id": RUN_ID,
                    }
                    response = await client.post(
                        f"{PERPLEXITY_URL}/chat/completions",
                        json=payload,
                    )
                    response.raise_for_status()
                    return response.json()

            result = await retry_with_backoff(perplexity_call)

            response_text = result["choices"][0]["message"]["content"]
            cost = result.get("cost", 0.0)
            TOTAL_COST += cost

            citations = result.get("citations", [])
            search_results = result.get("search_results", [])

            print(f"[FORECAST] Success with {model}")
            print(
                f"[FORECAST] Found {len(citations)} citations, {len(search_results)} search results"
            )
            print(f"[FORECAST] Cost: ${cost:.6f} | Total: ${TOTAL_COST:.6f}")

            prediction = 0.5
            reasoning = "No reasoning provided."

            for line in response_text.strip().split("\n"):
                if line.startswith("PREDICTION:"):
                    try:
                        pred_str = line.replace("PREDICTION:", "").strip()
                        prediction = clip_probability(float(pred_str))
                    except Exception:
                        pass
                elif line.startswith("REASONING:"):
                    reasoning = line.replace("REASONING:", "").strip()

            print(f"[FORECAST] Prediction: {prediction}")
            print(f"[FORECAST] Reasoning: {reasoning}")
            print(f"[FORECAST] Citations: {citations}")

            return {
                "event_id": event.event_id,
                "prediction": prediction,
                "reasoning": reasoning,
                "citations": citations,
                "search_results_count": len(search_results),
            }

        except Exception as e:
            print(f"[FORECAST] Error with {model}: {e}")
            if i == len(PERPLEXITY_MODELS) - 1:
                print("[FORECAST] All models failed. Returning fallback prediction.")
                return {
                    "event_id": event.event_id,
                    "prediction": 0.5,
                    "reasoning": "Unable to generate forecast. Returning neutral prediction.",
                }


async def run_agent(event: AgentData) -> dict:
    global TOTAL_COST
    TOTAL_COST = 0.0

    start_time = time.time()
    result = await forecast_with_perplexity(event)
    elapsed = time.time() - start_time

    print(f"[AGENT] Complete in {elapsed:.2f}s")
    print(f"[AGENT] Total run cost: ${TOTAL_COST:.6f}")

    return {
        "event_id": result["event_id"],
        "prediction": result["prediction"],
        "reasoning": result["reasoning"],
    }


def agent_main(event_data: dict) -> dict:
    event = AgentData.model_validate(event_data)
    print(f"\n[AGENT] Running forecast for event: {event.event_id}")
    print(f"[AGENT] Title: {event.title}")

    return asyncio.run(run_agent(event))
