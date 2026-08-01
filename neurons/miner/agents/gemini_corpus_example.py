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
OPENROUTER_URL = f"{PROXY_URL}/api/gateway/openrouter/chat/completions"
SIGNALS_URL = f"{PROXY_URL}/api/gateway/numinous-signals"

MODEL = "google/gemini-3.1-pro-preview"

MAX_SOURCES = 10
FETCH_TOP_N = 3
MAX_SOURCE_CHARS = 4000

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


async def corpus_search(query: str, cutoff: datetime) -> list[dict]:
    print(f"[CORPUS] Searching corpus: {query[:80]}")

    async def search_call():
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{SIGNALS_URL}/corpus/search",
                json={
                    "run_id": RUN_ID,
                    "query": query,
                    "max_results": MAX_SOURCES,
                    "published_before": cutoff.isoformat(),
                },
            )
            response.raise_for_status()
            return response.json()

    data = await retry_with_backoff(search_call)
    results = data.get("results", [])
    print(f"[CORPUS] Found {len(results)} sources")
    return results


async def corpus_fetch(source_id: str) -> dict | None:
    async def fetch_call():
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{SIGNALS_URL}/corpus/fetch",
                json={
                    "run_id": RUN_ID,
                    "source_id": source_id,
                },
            )
            response.raise_for_status()
            return response.json()

    try:
        return await retry_with_backoff(fetch_call)
    except Exception as e:
        print(f"[CORPUS] Fetch failed for {source_id}: {e}")
        return None


async def gather_research(event: AgentData) -> str:
    results = await corpus_search(event.title, event.cutoff)
    if not results:
        return "No corpus sources found for this event."

    sections = []
    for i, result in enumerate(results):
        title = result.get("title") or "Untitled"
        published = result.get("published_at") or "unknown date"

        if i < FETCH_TOP_N:
            fetched = await corpus_fetch(result["source_id"])
            if fetched and fetched.get("content"):
                body = fetched["content"][:MAX_SOURCE_CHARS]
            else:
                body = result.get("snippet", "")
        else:
            body = result.get("snippet", "")

        sections.append(f"[Source {i + 1}] {title} (published {published})\n{body}")

    return "\n\n".join(sections)


def build_forecast_messages(event: AgentData, research: str) -> list[dict]:
    cutoff_date = event.cutoff.strftime("%Y-%m-%d %H:%M UTC")

    system_prompt = """You are an expert forecaster specializing in probabilistic predictions.
Your task is to estimate the likelihood of binary events (YES/NO outcomes).

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

**Research from the Numinous corpus (time-stamped sources):**
{research}

**Your Task:**
Estimate the probability (0.0 to 1.0) that this event will occur or resolve as YES by the deadline.

Consider:
1. What the corpus sources say — weigh recency and reliability
2. Base rates and historical precedents for similar events
3. Current trends and momentum
4. Uncertainties and unknowns

**Required Output Format:**
PREDICTION: [number between 0.0 and 1.0]
REASONING: [2-4 sentences explaining your probability estimate, key evidence from the sources, and main uncertainties]"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


async def forecast_with_gemini(event: AgentData, research: str) -> dict:
    global TOTAL_COST
    print(f"[FORECAST] Generating forecast with {MODEL}...")

    messages = build_forecast_messages(event, research)

    try:

        async def openrouter_call():
            async with httpx.AsyncClient(timeout=120.0) as client:
                payload = {
                    "model": MODEL,
                    "messages": messages,
                    "temperature": 0.2,
                    "max_tokens": 1024,
                    "run_id": RUN_ID,
                }
                response = await client.post(
                    OPENROUTER_URL,
                    json=payload,
                )
                response.raise_for_status()
                return response.json()

        result = await retry_with_backoff(openrouter_call)

        response_text = result["choices"][0]["message"]["content"]
        cost = result.get("cost", 0.0)
        TOTAL_COST += cost

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

        return {
            "event_id": event.event_id,
            "prediction": prediction,
            "reasoning": reasoning,
        }

    except Exception as e:
        print(f"[FORECAST] Error with {MODEL}: {e}")
        return {
            "event_id": event.event_id,
            "prediction": 0.5,
            "reasoning": "Unable to generate forecast. Returning neutral prediction.",
        }


async def run_agent(event: AgentData) -> dict:
    global TOTAL_COST
    TOTAL_COST = 0.0

    start_time = time.time()

    try:
        research = await gather_research(event)
    except Exception as e:
        print(f"[CORPUS] Research failed: {e}")
        research = "No corpus sources available for this event."

    result = await forecast_with_gemini(event, research)
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
