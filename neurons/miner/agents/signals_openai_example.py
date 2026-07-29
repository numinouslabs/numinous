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
OPENAI_URL = f"{PROXY_URL}/api/gateway/openai"

OPENAI_MODELS = [
    "gpt-5.2",
    "gpt-5",
    "gpt-5-mini",
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
                    raise Exception(
                        f"Rate limit exceeded after {max_retries} retries: {error_detail}"
                    )
            else:
                raise Exception(f"HTTP {e.response.status_code}: {error_detail}")
        except Exception:
            raise


def clip_probability(prediction: float) -> float:
    return max(0.0, min(1.0, prediction))


# =============================================================================
# PHASE 1: NUMINOUS SIGNALS
# =============================================================================


def format_signals(signals: list[dict]) -> str:
    if not signals:
        return ""
    lines = []
    for s in signals:
        direction = s.get("direction", "neutral")
        relevance = s.get("relevance_score", 0)
        impact = s.get("impact_score", 0)
        lines.append(
            f"- [{direction}] {s.get('headline', '')} "
            f"(relevance={relevance:.2f}, impact={impact:.2f}, source={s.get('source', '')})"
        )
    return "\n".join(lines)


async def fetch_signals(event: AgentData) -> str:
    global TOTAL_COST
    print("[SIGNALS] Fetching relevant signals...")

    market_url = (event.metadata or {}).get("condition_id")
    payload = {
        "run_id": RUN_ID,
        "max_events_per_source": 10,
        "time_window_hours": 72,
    }
    if market_url:
        payload["market"] = market_url
    else:
        payload["question"] = event.title

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(f"{SIGNALS_URL}/signals", json=payload)
            response.raise_for_status()
            data = response.json()

            signals = data.get("signals", [])
            cost = data.get("cost", 0.0)
            TOTAL_COST += cost

            total = data.get("total_event_count", 0)
            filtered = data.get("filtered_count", 0)
            failed = data.get("failed_sources", [])

            print(
                f"[SIGNALS] Got {len(signals)} signals "
                f"(total={total}, filtered={filtered}, failed={failed}, cost=${cost:.6f})"
            )

            return format_signals(signals)

        except Exception as e:
            print(f"[SIGNALS] Failed: {e}")
            return ""


# =============================================================================
# PHASE 2: OPENAI INFERENCE FORECAST
# =============================================================================


def build_forecast_messages(event: AgentData, signals_context: str) -> list[dict]:
    cutoff_date = event.cutoff.strftime("%Y-%m-%d %H:%M UTC")

    signals_section = ""
    if signals_context:
        signals_section = f"""

**Relevant Signals (from Numinous Signals API):**
{signals_context}
"""

    return [
        {
            "role": "developer",
            "content": (
                "You are an expert forecaster. "
                "You have access to curated news signals. "
                "Use them to make well-calibrated probabilistic predictions."
            ),
        },
        {
            "role": "user",
            "content": f"""**Event to Forecast:**
{event.title}

**Description:**
{event.description}

**Deadline:** {cutoff_date}
{signals_section}
Weigh the signals above against the time remaining, then estimate the probability (0.0 to 1.0) that this event resolves YES.

**Required Output Format:**
PREDICTION: [number between 0.0 and 1.0]
REASONING: [2-4 sentences explaining your estimate]""",
        },
    ]


def extract_openai_response_text(response_data: dict) -> str:
    output = response_data.get("output", [])
    if not output or not isinstance(output, list):
        return ""

    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            text = content.get("text", "")
            if text and content.get("type") in ("output_text", "text"):
                return text

    return ""


async def call_openai(model: str, messages: list[dict]) -> tuple[str, float]:
    async with httpx.AsyncClient(timeout=120.0) as client:
        payload = {
            "model": model,
            "input": messages,
            "run_id": RUN_ID,
        }
        response = await client.post(f"{OPENAI_URL}/responses/inference", json=payload)
        response.raise_for_status()

        data = response.json()
        return extract_openai_response_text(data), data.get("cost", 0.0)


def parse_llm_response(response_text: str) -> tuple[float, str]:
    try:
        prediction = 0.5
        reasoning = "No reasoning provided."

        for line in response_text.strip().split("\n"):
            if line.startswith("PREDICTION:"):
                prediction = clip_probability(float(line.replace("PREDICTION:", "").strip()))
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()

        return prediction, reasoning
    except Exception as e:
        print(f"[WARNING] Failed to parse LLM response: {e}")
        return 0.5, "Failed to parse LLM response."


async def forecast_with_signals(event: AgentData, signals_context: str) -> dict:
    global TOTAL_COST
    print("[FORECAST] Generating forecast with OpenAI inference...")

    messages = build_forecast_messages(event, signals_context)

    for i, model in enumerate(OPENAI_MODELS):
        print(f"[FORECAST] Trying model {i+1}/{len(OPENAI_MODELS)}: {model}")

        try:

            async def llm_call():
                return await call_openai(model, messages)

            response_text, cost = await retry_with_backoff(llm_call)
            TOTAL_COST += cost
            prediction, reasoning = parse_llm_response(response_text)

            print(f"[FORECAST] Success with {model}: prediction={prediction}")
            print(f"[FORECAST] Cost: ${cost:.6f} | Total: ${TOTAL_COST:.6f}")
            return {
                "event_id": event.event_id,
                "prediction": prediction,
                "reasoning": reasoning,
            }

        except httpx.HTTPStatusError as e:
            try:
                error_detail = e.response.json().get("detail", "")
            except Exception:
                error_detail = e.response.text[:200] if hasattr(e.response, "text") else ""

            detail_msg = f": {error_detail}" if error_detail else ""
            print(
                f"[FORECAST] HTTP error {e.response.status_code} with {model}{detail_msg}. "
                f"Trying next model..."
            )

        except Exception as e:
            print(f"[FORECAST] Error with {model}: {e}. Trying next model...")

    print("[FORECAST] All models failed. Returning fallback prediction.")
    return {
        "event_id": event.event_id,
        "prediction": 0.5,
        "reasoning": "All models failed. Returning neutral prediction.",
    }


# =============================================================================
# MAIN AGENT
# =============================================================================


async def run_agent(event: AgentData) -> dict:
    global TOTAL_COST
    TOTAL_COST = 0.0

    start_time = time.time()

    signals_context = await fetch_signals(event)
    result = await forecast_with_signals(event, signals_context)

    elapsed = time.time() - start_time
    print(f"[AGENT] Complete in {elapsed:.2f}s")
    print(f"[AGENT] Total run cost: ${TOTAL_COST:.6f}")

    return result


def agent_main(event_data: dict) -> dict:
    event = AgentData.model_validate(event_data)
    print(f"\n[AGENT] Running Signals + OpenAI forecast for event: {event.event_id}")
    print(f"[AGENT] Title: {event.title}")

    return asyncio.run(run_agent(event))
