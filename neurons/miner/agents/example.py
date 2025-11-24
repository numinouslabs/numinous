import asyncio
import os
import time
from datetime import datetime

import httpx
from pydantic import BaseModel

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Fetch run ID from environment
RUN_ID = os.getenv("RUN_ID")
if not RUN_ID:
    raise ValueError("RUN_ID environment variable is required but not set")

# Fetch proxy URL from environment
PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
CHUTES_URL = f"{PROXY_URL}/api/gateway/chutes"
DESEARCH_URL = f"{PROXY_URL}/api/gateway/desearch"


# =============================================================================
# CONSTANTS
# =============================================================================

MIN_INSTANCES = 5
LLMS = [
    "tngtech/DeepSeek-TNG-R1T2-Chimera",  # 671B Tri-Mind (V3+R1+R1-0528 hybrid), fixed <think> token
    "deepseek-ai/DeepSeek-V3.1",  # 685B MoE, general-purpose powerhouse
    "zai-org/GLM-4.5",  # Faster tool-calling specialist
    "openai/gpt-oss-120b",  # 120B open-source fallback
]

# Retry configuration
MAX_RETRIES = 3
BASE_BACKOFF = 1.5  # seconds


# =============================================================================
# MODELS
# =============================================================================


class AgentData(BaseModel):
    event_id: str
    title: str
    description: str
    cutoff: datetime
    metadata: dict


class ChuteModelStatus(BaseModel):
    chute_id: str
    name: str
    active_instance_count: int


# =============================================================================
# TOOL PROMPTS
# =============================================================================


def build_research_prompt(event: AgentData) -> str:
    """Build targeted research prompt for Desearch."""
    return f"""Search for recent information to help forecast this event:
"{event.title}"

Focus on:
- Latest news, announcements, or developments related to this topic
- Historical patterns or precedents
- Expert opinions or market sentiment
- Any relevant data, statistics, or indicators

Event description: {event.description}
Forecast deadline: {event.cutoff.strftime('%Y-%m-%d')}"""


def build_forecast_messages(event: AgentData, context: str) -> list[dict]:
    """Build LLM messages for forecasting."""
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

**Research Context:**
{context if context else "No additional research context available. Base your forecast on the event description and general knowledge."}

**Your Task:**
Estimate the probability (0.0 to 1.0) that this event will occur or resolve as YES by the deadline.

Consider:
1. What is the base rate for similar events?
2. What specific evidence supports or contradicts this outcome?
3. What uncertainties or unknowns remain?
4. How confident are you in available information?

**Required Output Format:**
PREDICTION: [number between 0.0 and 1.0]
REASONING: [2-4 sentences explaining your probability estimate, key factors considered, and main uncertainties]"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


async def fetch_chutes_active_models(min_instances: int = 5) -> list[ChuteModelStatus]:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{CHUTES_URL}/status")
            response.raise_for_status()

            chutes_statuses = [ChuteModelStatus.model_validate(item) for item in response.json()]
            filtered_chutes = [
                chute for chute in chutes_statuses if chute.active_instance_count >= min_instances
            ]
            return filtered_chutes

    except Exception as e:
        print(f"[WARNING] Failed to fetch chutes status: {e}")
        return []


def get_available_models(all_models: list[str], active_chutes: list[ChuteModelStatus]) -> list[str]:
    active_names = {chute.name for chute in active_chutes}
    available = [model for model in all_models if model in active_names]

    if available:
        print(f"[INFO] Available models: {available}")
    else:
        print("[WARNING] No preferred models available. Will try all from list.")
        available = all_models  # Fallback: try anyway

    return available


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

            if e.response.status_code == 429:  # Rate limit
                if attempt < max_retries - 1:
                    delay = BASE_BACKOFF ** (attempt + 1)
                    print(f"[RETRY] Rate limited (429), retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    raise Exception(
                        f"Rate limit exceeded after {max_retries} retries: {error_detail}"
                    )
            else:
                # Don't retry other HTTP errors - re-raise with detail
                raise Exception(f"HTTP {e.response.status_code}: {error_detail}")
        except Exception:
            # Unknown error - don't retry
            raise


def clip_probability(prediction: float) -> float:
    return max(0.0, min(1.0, prediction))


# =============================================================================
# PHASE 1: RESEARCH WITH DESEARCH
# =============================================================================


async def research_event(event: AgentData) -> str:
    print("[PHASE 1] Researching event via Desearch...")

    try:

        async def desearch_call():
            async with httpx.AsyncClient(timeout=30.0) as client:
                payload = {
                    "prompt": build_research_prompt(event),
                    "model": "NOVA",
                    "tools": ["web", "reddit", "wikipedia"],
                    "count": 5,
                    "run_id": RUN_ID,
                }
                response = await client.post(f"{DESEARCH_URL}/ai/search", json=payload)
                response.raise_for_status()
                return response.json()

        result = await retry_with_backoff(desearch_call)

        # Extract context from response
        context = result.get("completion", "")
        if context:
            context = context[:5000]  # Limit size
            # Show preview of research findings
            preview = context[:300].replace("\n", " ")
            print(f"[PHASE 1] Research complete. Context length: {len(context)}")
            print(f"[PHASE 1] Preview: {preview}...")
        else:
            print("[PHASE 1] No context in response")

        return context

    except Exception as e:
        print(f"[PHASE 1] Research failed: {e}. Continuing without context.")
        return ""


# =============================================================================
# PHASE 2: FORECAST WITH LLM (CHUTES)
# =============================================================================
#
# Error Handling Strategy for 503 (Service Unavailable):
#
# 503 can mean two things:
# 1. No instances available (cold model) - retrying won't help immediately
# 2. Model overloaded/restarting - short retry might work
#
# Solution: Try SHORT retry (2 attempts, 1.5s backoff) before swapping models
# This handles overloaded scenarios without wasting too much time on cold models


async def call_llm(model: str, messages: list[dict]) -> str:
    async with httpx.AsyncClient(timeout=45.0) as client:
        payload = {
            "model": model,
            "messages": messages,
            "run_id": RUN_ID,
        }  # Must ALWAYS include RUN_ID on body
        response = await client.post(
            f"{CHUTES_URL}/chat/completions",
            json=payload,
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]


def parse_llm_response(response_text: str) -> tuple[float, str]:
    try:
        lines = response_text.strip().split("\n")
        prediction = 0.5
        reasoning = "No reasoning provided."

        for line in lines:
            if line.startswith("PREDICTION:"):
                # Clip probability to [0.0, 1.0]
                prediction = clip_probability(float(line.replace("PREDICTION:", "").strip()))
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()

        return prediction, reasoning

    except Exception as e:
        print(f"[WARNING] Failed to parse LLM response: {e}")
        return 0.5, "Failed to parse LLM response."


async def forecast_with_llm(event: AgentData, context: str, available_models: list[str]) -> dict:
    print("[PHASE 2] Generating forecast with LLM...")

    messages = build_forecast_messages(event, context)

    # Try each available model in order
    for i, model in enumerate(available_models):
        print(f"[PHASE 2] Trying model {i+1}/{len(available_models)}: {model}")

        max_503_retries = 2
        backoff_503 = 1.5

        for attempt in range(max_503_retries):
            try:

                async def llm_call():
                    return await call_llm(model, messages)

                response_text = await retry_with_backoff(llm_call)
                prediction, reasoning = parse_llm_response(response_text)

                print(f"[PHASE 2] Success with {model}: prediction={prediction}")
                return {
                    "event_id": event.event_id,
                    "prediction": prediction,
                    "reasoning": reasoning,
                }

            except httpx.HTTPStatusError as e:
                try:
                    error_detail = e.response.json().get("detail", "")
                except Exception:
                    error_detail = ""

                if e.response.status_code == 503:
                    if attempt < max_503_retries - 1:
                        delay = backoff_503 ** (attempt + 1)
                        print(f"[PHASE 2] Model {model} unavailable (503). Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                        continue  # Retry same model
                    else:
                        # After retries, swap to next model
                        detail_msg = f": {error_detail}" if error_detail else ""
                        print(
                            f"[PHASE 2] Model {model} still unavailable after {max_503_retries} retries{detail_msg}. Trying next model..."
                        )
                        break
                else:
                    detail_msg = f": {error_detail}" if error_detail else ""
                    print(
                        f"[PHASE 2] HTTP error {e.response.status_code} with {model}{detail_msg}. Trying next model..."
                    )
                    break

            except Exception as e:
                print(f"[PHASE 2] Error with {model}: {e}. Trying next model...")
                break

    print("[PHASE 2] All models failed. Returning fallback prediction.")
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
    Two-phase forecasting agent:
    1. Research: Gather context using Desearch
    2. Forecast: Generate prediction using LLM (Chutes)

    Demonstrates:
    - Model availability checking
    - Retry with exponential backoff
    - Model swapping on 503 errors
    - Graceful fallback
    """
    start_time = time.time()

    # Check which models are available
    active_chutes = await fetch_chutes_active_models(min_instances=MIN_INSTANCES)
    available_models = get_available_models(LLMS, active_chutes)

    if not available_models:
        print("[WARNING] No models available. Will attempt with preferred list anyway.")
        available_models = LLMS

    # Phase 1: Research
    context = await research_event(event)

    # Phase 2: Forecast
    result = await forecast_with_llm(event, context, available_models)

    elapsed = time.time() - start_time
    print(f"[AGENT] Complete in {elapsed:.2f}s")

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
