import asyncio
import os
import time
from datetime import datetime

import httpx
from pydantic import BaseModel

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

RUN_ID = os.getenv("RUN_ID")
if not RUN_ID:
    raise ValueError("RUN_ID environment variable is required but not set")

PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
OPENAI_URL = f"{PROXY_URL}/api/gateway/openai"


# =============================================================================
# CONSTANTS
# =============================================================================

OPENAI_MODELS = [
    "gpt-5.2",
    "gpt-5",
    "gpt-5-mini",
]

MAX_RETRIES = 3
BASE_BACKOFF = 1.5

TOTAL_COST = 0.0


# =============================================================================
# MODELS
# =============================================================================


class AgentData(BaseModel):
    event_id: str
    title: str
    description: str
    cutoff: datetime
    metadata: dict


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


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
# OPENAI WEB SEARCH FORECASTING
# =============================================================================


def build_forecast_messages(event: AgentData) -> list[dict]:
    cutoff_date = event.cutoff.strftime("%Y-%m-%d %H:%M UTC")

    system_prompt = """You are an expert forecaster specializing in probabilistic predictions.
Your task is to estimate the likelihood of binary events (YES/NO outcomes).

You have access to web search. Use it to research current information before making your forecast.

Key principles:
- Search for recent news, data, and expert opinions
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
Before making your forecast, systematically research:
1. Search for recent news and developments related to this event
2. Search for expert analysis, predictions, or market sentiment
3. Search for historical data or precedents for similar events

Then estimate the probability (0.0 to 1.0) that this event will occur or resolve as YES by the deadline.

Consider:
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


def convert_messages_to_openai_input(messages: list[dict]) -> list[dict]:
    openai_input = []
    for msg in messages:
        role = msg["role"]
        if role == "system":
            role = "developer"
        openai_input.append({"role": role, "content": msg["content"]})
    return openai_input


def extract_openai_response_text(response_data: dict) -> str:
    if not response_data:
        return ""

    output = response_data.get("output", [])
    if not output or not isinstance(output, list):
        return ""

    for item in output:
        if not item or not isinstance(item, dict):
            continue

        if item.get("type") == "message":
            content_list = item.get("content")
            if not content_list or not isinstance(content_list, list):
                continue

            for content in content_list:
                if not content or not isinstance(content, dict):
                    continue

                if content.get("type") == "output_text" and content.get("text"):
                    return content.get("text")
                elif content.get("type") == "text":
                    text_val = content.get("text", "")
                    if text_val:
                        return text_val

    return ""


async def call_openai_llm(model: str, messages: list[dict]) -> tuple[str, float]:
    async with httpx.AsyncClient(timeout=120.0) as client:
        openai_input = convert_messages_to_openai_input(messages)
        payload = {
            "model": model,
            "input": openai_input,
            "tools": [{"type": "web_search"}],
            "run_id": RUN_ID,
        }

        url = f"{OPENAI_URL}/responses"
        response = await client.post(url, json=payload)
        response.raise_for_status()

        data = response.json()
        content = extract_openai_response_text(data)
        cost = data.get("cost", 0.0)
        return content, cost


def parse_llm_response(response_text: str) -> tuple[float, str]:
    try:
        lines = response_text.strip().split("\n")
        prediction = 0.5
        reasoning = "No reasoning provided."

        for line in lines:
            if line.startswith("PREDICTION:"):
                prediction = clip_probability(float(line.replace("PREDICTION:", "").strip()))
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()

        return prediction, reasoning

    except Exception as e:
        print(f"[WARNING] Failed to parse LLM response: {e}")
        return 0.5, "Failed to parse LLM response."


async def forecast_with_websearch(event: AgentData) -> dict:
    global TOTAL_COST
    print("[FORECAST] Generating forecast with OpenAI web search...")

    messages = build_forecast_messages(event)

    for i, model in enumerate(OPENAI_MODELS):
        print(f"[FORECAST] Trying model {i+1}/{len(OPENAI_MODELS)}: {model}")

        try:

            async def llm_call():
                return await call_openai_llm(model, messages)

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
                f"[FORECAST] HTTP error {e.response.status_code} with {model}{detail_msg}. Trying next model..."
            )

        except Exception as e:
            print(f"[FORECAST] Error with {model}: {e}. Trying next model...")

    print("[FORECAST] All models failed. Returning fallback prediction.")
    return {
        "event_id": event.event_id,
        "prediction": 0.5,
        "reasoning": "Unable to generate forecast due to model availability issues. Returning neutral prediction.",
    }


# =============================================================================
# MAIN AGENT
# =============================================================================


async def run_agent(event: AgentData) -> dict:
    """
    Single-phase forecasting agent with OpenAI web search.

    Demonstrates:
    - OpenAI web_search tool integration
    - Retry with exponential backoff
    - Model fallback on errors
    - Web search cost tracking
    """
    global TOTAL_COST
    TOTAL_COST = 0.0

    start_time = time.time()

    result = await forecast_with_websearch(event)

    elapsed = time.time() - start_time
    print(f"[AGENT] Complete in {elapsed:.2f}s")
    print(f"[AGENT] Total run cost: ${TOTAL_COST:.6f}")

    return result


def agent_main(event_data: dict) -> dict:
    """
    Entry point for the forecasting agent.

    Args:
        event_data: Event information dict

    Returns:
        dict with keys: event_id, prediction, reasoning
    """
    event = AgentData.model_validate(event_data)
    print(f"\n[AGENT] Running forecast for event: {event.event_id}")
    print(f"[AGENT] Title: {event.title}")

    return asyncio.run(run_agent(event))
