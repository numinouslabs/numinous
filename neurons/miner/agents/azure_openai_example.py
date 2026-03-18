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
AZURE_OPENAI_URL = f"{PROXY_URL}/api/gateway/azure-openai"
DESEARCH_URL = f"{PROXY_URL}/api/gateway/desearch"


# =============================================================================
# CONSTANTS
# =============================================================================

# Azure OpenAI deployment names (these are YOUR deployment names in Azure)
# Replace these with your actual Azure OpenAI deployment names
AZURE_DEPLOYMENTS = [
    "gpt-5-2-deployment",      # Your GPT-5.2 deployment
    "gpt-5-mini-deployment",   # Your GPT-5-mini deployment
    "gpt-5-deployment",        # Your GPT-5 deployment
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
# TOOL PROMPTS
# =============================================================================


def build_research_prompt(event: AgentData) -> str:
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


async def retry_with_backoff(func, max_retries: int = MAX_RETRIES):
    """
    Retry a function with exponential backoff.
    
    Handles:
    - Timeout errors (retries with backoff)
    - Rate limiting (429 errors)
    - Other HTTP errors (raises immediately)
    """
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
    """Ensure prediction is in valid range [0.0, 1.0]"""
    return max(0.0, min(1.0, prediction))


# =============================================================================
# PHASE 1: RESEARCH WITH DESEARCH
# =============================================================================


async def research_event(event: AgentData) -> str:
    """
    Phase 1: Research the event using Desearch.
    
    Returns context string (up to 5000 chars) for the forecasting phase.
    """
    global TOTAL_COST
    print("[PHASE 1] Researching event via Desearch...")

    try:

        async def desearch_call():
            async with httpx.AsyncClient(timeout=30.0) as client:
                payload = {
                    "prompt": build_research_prompt(event),
                    "model": "NOVA",
                    "tools": ["web", "reddit", "wikipedia"],
                    "count": 10,
                    "run_id": RUN_ID,
                }
                response = await client.post(f"{DESEARCH_URL}/ai/search", json=payload)
                response.raise_for_status()
                return response.json()

        result = await retry_with_backoff(desearch_call)

        context = result.get("completion", "")
        cost = result.get("cost", 0.0)
        TOTAL_COST += cost

        if context:
            context = context[:5000]
            preview = context[:300].replace("\n", " ")
            print(f"[PHASE 1] Research complete. Context length: {len(context)}")
            print(f"[PHASE 1] Preview: {preview}...")
        else:
            print("[PHASE 1] No context in response")

        print(f"[PHASE 1] Cost: ${cost:.6f} | Total: ${TOTAL_COST:.6f}")

        return context

    except Exception as e:
        print(f"[PHASE 1] Research failed: {e}. Continuing without context.")
        return ""


# =============================================================================
# PHASE 2: FORECAST WITH AZURE OPENAI
# =============================================================================


def convert_messages_to_azure_input(messages: list[dict]) -> list[dict]:
    """
    Convert standard chat messages to Azure OpenAI input format.
    
    Azure OpenAI uses 'developer' role instead of 'system' role.
    This matches the OpenAI API format.
    """
    azure_input = []
    for msg in messages:
        role = msg["role"]
        if role == "system":
            role = "developer"
        azure_input.append({"role": role, "content": msg["content"]})
    return azure_input


def extract_azure_response_text(response_data: dict) -> str:
    """
    Extract text content from Azure OpenAI response.
    
    Handles both 'output_text' and 'text' content types.
    Returns empty string if no text found.
    """
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


async def call_azure_openai_llm(deployment: str, messages: list[dict]) -> tuple[str, float]:
    """
    Call Azure OpenAI via the gateway.
    
    Args:
        deployment: Azure deployment name (not model name!)
        messages: List of chat messages
        
    Returns:
        tuple of (response_text, cost)
        
    Note: The key difference from OpenAI is using 'deployment' instead of 'model'
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        azure_input = convert_messages_to_azure_input(messages)
        payload = {
            "deployment": deployment,  # ← KEY DIFFERENCE: 'deployment' not 'model'
            "input": azure_input,
            "run_id": RUN_ID,
        }

        url = f"{AZURE_OPENAI_URL}/responses"
        response = await client.post(url, json=payload)
        response.raise_for_status()

        data = response.json()
        content = extract_azure_response_text(data)
        cost = data.get("cost", 0.0)
        return content, cost


def parse_llm_response(response_text: str) -> tuple[float, str]:
    """
    Parse LLM response to extract prediction and reasoning.
    
    Expected format:
        PREDICTION: 0.75
        REASONING: The event is likely because...
        
    Returns:
        tuple of (prediction, reasoning)
    """
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


async def forecast_with_azure_llm(event: AgentData, context: str) -> dict:
    """
    Phase 2: Generate forecast using Azure OpenAI with deployment fallback.
    
    Tries each deployment in AZURE_DEPLOYMENTS list until one succeeds.
    Returns fallback prediction (0.5) if all deployments fail.
    
    This demonstrates:
    - Azure OpenAI deployment-based routing
    - Automatic fallback between deployments
    - Cost tracking across multiple attempts
    - Robust error handling
    """
    global TOTAL_COST
    print("[PHASE 2] Generating forecast with Azure OpenAI...")

    messages = build_forecast_messages(event, context)

    for i, deployment in enumerate(AZURE_DEPLOYMENTS):
        print(f"[PHASE 2] Trying deployment {i+1}/{len(AZURE_DEPLOYMENTS)}: {deployment}")

        try:

            async def llm_call():
                return await call_azure_openai_llm(deployment, messages)

            response_text, cost = await retry_with_backoff(llm_call)
            TOTAL_COST += cost
            prediction, reasoning = parse_llm_response(response_text)

            print(f"[PHASE 2] Success with {deployment}: prediction={prediction}")
            print(f"[PHASE 2] Cost: ${cost:.6f} | Total: ${TOTAL_COST:.6f}")
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
                f"[PHASE 2] HTTP error {e.response.status_code} with {deployment}{detail_msg}. Trying next deployment..."
            )

        except Exception as e:
            print(f"[PHASE 2] Error with {deployment}: {e}. Trying next deployment...")

    print("[PHASE 2] All deployments failed. Returning fallback prediction.")
    return {
        "event_id": event.event_id,
        "prediction": 0.5,
        "reasoning": "Unable to generate forecast due to deployment availability issues. Returning neutral prediction.",
    }


# =============================================================================
# MAIN AGENT
# =============================================================================


async def run_agent(event: AgentData) -> dict:
    """
    Two-phase forecasting agent using Azure OpenAI.
    
    Phase 1: Research - Gather context using Desearch
    Phase 2: Forecast - Generate prediction using Azure OpenAI
    
    Demonstrates:
    - Azure OpenAI API integration via gateway
    - Deployment-based routing (key difference from OpenAI)
    - Retry with exponential backoff
    - Deployment fallback on errors
    - Cost tracking across both phases
    
    Returns:
        dict with keys: event_id, prediction, reasoning
    """
    global TOTAL_COST
    TOTAL_COST = 0.0

    start_time = time.time()

    context = await research_event(event)
    result = await forecast_with_azure_llm(event, context)

    elapsed = time.time() - start_time
    print(f"[AGENT] Complete in {elapsed:.2f}s")
    print(f"[AGENT] Total run cost: ${TOTAL_COST:.6f}")

    return result


def agent_main(event_data: dict) -> dict:
    """
    Entry point for the forecasting agent using Azure OpenAI.
    
    Args:
        event_data: Event information dict with keys:
            - event_id: str
            - title: str
            - description: str
            - cutoff: datetime
            - metadata: dict
    
    Returns:
        dict with keys:
            - event_id: str
            - prediction: float (0.0 to 1.0)
            - reasoning: str
    
    Example:
        event = {
            "event_id": "evt_123",
            "title": "Will X happen by Y date?",
            "description": "Full event description...",
            "cutoff": datetime(2026, 12, 31),
            "metadata": {}
        }
        result = agent_main(event)
        print(f"Prediction: {result['prediction']}")
        print(f"Reasoning: {result['reasoning']}")
    """
    event = AgentData.model_validate(event_data)
    print(f"\n[AGENT] Running forecast for event: {event.event_id}")
    print(f"[AGENT] Title: {event.title}")

    return asyncio.run(run_agent(event))


# =============================================================================
# AZURE OPENAI SPECIFIC EXAMPLES
# =============================================================================


async def example_basic_call():
    print("\n=== Example 1: Basic Azure OpenAI Call ===\n")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        payload = {
            "deployment": "gpt-5-mini-deployment",  # Your deployment name
            "input": [
                {"role": "developer", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"}
            ],
            "run_id": RUN_ID,
            "temperature": 0.7,
        }
        
        response = await client.post(f"{AZURE_OPENAI_URL}/responses", json=payload)
        response.raise_for_status()
        
        data = response.json()
        text = extract_azure_response_text(data)
        cost = data.get("cost", 0.0)
        
        print(f"Response: {text}")
        print(f"Cost: ${cost:.6f}")


async def example_with_parameters():
    print("\n=== Example 2: Advanced Parameters ===\n")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        payload = {
            "deployment": "gpt-5-deployment",
            "input": [
                {"role": "user", "content": "Search for recent news about AI."}
            ],
            "run_id": RUN_ID,
            "temperature": 0.3,  # Lower temperature for more focused responses
            "max_output_tokens": 500,  # Limit response length
            "tools": [{"type": "web_search"}],  # Enable web search
        }
        
        response = await client.post(f"{AZURE_OPENAI_URL}/responses", json=payload)
        response.raise_for_status()
        
        data = response.json()
        print(f"Full response structure:")
        print(f"- ID: {data.get('id')}")
        print(f"- Model: {data.get('model')}")
        print(f"- Output items: {len(data.get('output', []))}")
        print(f"- Usage: {data.get('usage')}")
        print(f"- Cost: ${data.get('cost', 0.0):.6f}")


async def example_deployment_fallback():
    print("\n=== Example 3: Deployment Fallback ===\n")
    
    deployments = ["gpt-5-2-deployment", "gpt-5-deployment", "gpt-5-mini-deployment"]
    
    for deployment in deployments:
        print(f"Trying deployment: {deployment}")
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                payload = {
                    "deployment": deployment,
                    "input": [
                        {"role": "user", "content": "Say hello"}
                    ],
                    "run_id": RUN_ID,
                }
                
                response = await client.post(f"{AZURE_OPENAI_URL}/responses", json=payload)
                response.raise_for_status()
                
                data = response.json()
                text = extract_azure_response_text(data)
                print(f"✓ Success with {deployment}")
                print(f"  Response: {text[:100]}...")
                break
                
        except Exception as e:
            print(f"✗ Failed with {deployment}: {e}")
            print(f"  Trying next deployment...")
    else:
        print("All deployments failed!")


async def run_examples():
    await example_basic_call()
    await example_with_parameters()
    await example_deployment_fallback()


if __name__ == "__main__":
    print("Azure OpenAI Integration Examples")
    print("=" * 50)
    asyncio.run(run_examples())