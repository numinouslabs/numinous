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
OPENAI_URL = f"{PROXY_URL}/api/gateway/openai"
INDICIA_URL = f"{PROXY_URL}/api/gateway/numinous-indicia"

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
# PHASE 1: INDICIA SIGNALS
# =============================================================================


def format_signals(signals: list[dict]) -> str:
    if not signals:
        return ""
    lines = []
    for s in signals:
        ts = s.get("timestamp", "")
        lines.append(
            f"- [{s.get('category', '')}] {s.get('signal', '')} "
            f"(confidence={s.get('confidence', '')}, status={s.get('fact_status', '')}, {ts})"
        )
    return "\n".join(lines)


async def fetch_indicia_signals() -> str:
    global TOTAL_COST
    print("[INDICIA] Fetching geopolitical signals...")

    all_signals: list[dict] = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        for endpoint, label in [("/x-osint", "X-OSINT"), ("/liveuamap", "LiveUAMap")]:
            try:
                payload = {"run_id": RUN_ID, "limit": 20}
                response = await client.post(f"{INDICIA_URL}{endpoint}", json=payload)
                response.raise_for_status()
                data = response.json()

                signals = data.get("signals", [])
                cost = data.get("cost", 0.0)
                TOTAL_COST += cost

                all_signals.extend(signals)
                print(f"[INDICIA] {label}: {len(signals)} signals (cost=${cost:.6f})")

            except Exception as e:
                print(f"[INDICIA] {label} failed: {e}")

    formatted = format_signals(all_signals)
    if formatted:
        print(f"[INDICIA] Total signals collected: {len(all_signals)}")
    else:
        print("[INDICIA] No signals retrieved")

    return formatted


# =============================================================================
# PHASE 2: OPENAI INFERENCE FORECAST
# =============================================================================


def build_forecast_messages(event: AgentData, indicia_context: str) -> list[dict]:
    cutoff_date = event.cutoff.strftime("%Y-%m-%d %H:%M UTC")

    indicia_section = ""
    if indicia_context:
        indicia_section = f"""

**Geopolitical Intelligence Signals (from Indicia):**
{indicia_context}
"""

    return [
        {
            "role": "developer",
            "content": (
                "You are an expert geopolitical forecaster. "
                "You have access to curated intelligence signals. "
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
{indicia_section}
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


async def forecast_with_signals(event: AgentData, indicia_context: str) -> dict:
    global TOTAL_COST
    print("[FORECAST] Generating forecast with OpenAI inference...")

    messages = build_forecast_messages(event, indicia_context)

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

    indicia_context = await fetch_indicia_signals()
    result = await forecast_with_signals(event, indicia_context)

    elapsed = time.time() - start_time
    print(f"[AGENT] Complete in {elapsed:.2f}s")
    print(f"[AGENT] Total run cost: ${TOTAL_COST:.6f}")

    return result


def agent_main(event_data: dict) -> dict:
    event = AgentData.model_validate(event_data)
    print(f"\n[AGENT] Running Indicia + OpenAI forecast for event: {event.event_id}")
    print(f"[AGENT] Title: {event.title}")

    return asyncio.run(run_agent(event))
