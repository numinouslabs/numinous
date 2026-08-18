import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, status

from neurons.miner.gateway.cache import cached_gateway_call
from neurons.miner.gateway.error_handler import handle_provider_errors
from neurons.miner.gateway.providers.lightning_rod import LightningRodClient
from neurons.miner.gateway.providers.numinous_indicia import NuminousIndiciaClient
from neurons.miner.gateway.providers.numinous_signals import NuminousSignalsClient
from neurons.miner.gateway.providers.openai import OpenAIClient
from neurons.miner.gateway.providers.openrouter import OpenRouterClient
from neurons.validator.models import numinous_client as models
from neurons.validator.models.lightning_rod import calculate_cost as calculate_lightning_rod_cost
from neurons.validator.models.numinous_indicia import (
    calculate_cost as calculate_numinous_indicia_cost,
)
from neurons.validator.models.numinous_signals import (
    NewsFeedArticle,
    calculate_causal_drivers_cost,
    calculate_corpus_fetch_cost,
    calculate_corpus_search_cost,
    calculate_deep_research_cost,
    calculate_news_feed_cost,
)
from neurons.validator.models.openai import calculate_cost as calculate_openai_cost
from neurons.validator.models.openrouter import calculate_cost as calculate_openrouter_cost

logger = logging.getLogger(__name__)

env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    logger.warning(
        "No .env file found. Make sure to create a .env file in the gateway/ directory "
        "and add your API keys."
    )


app = FastAPI(title="Numinous API Gateway")
gateway_router = APIRouter(prefix="/api/gateway")


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "API Gateway"}


_OPENAI_BUILTIN_TOOL_TYPES = frozenset(
    {
        "web_search",
        "web_search_preview",
        "file_search",
        "code_interpreter",
        "image_generation",
        "computer",
        "computer_use_preview",
        "tool_search",
    }
)


@gateway_router.post("/openai/responses/inference", response_model=models.GatewayOpenAIResponse)
@cached_gateway_call
@handle_provider_errors("OpenAI")
async def openai_create_inference_only_response(
    request: models.OpenAIInferenceRequest,
) -> models.GatewayOpenAIResponse:
    if request.tools:
        builtin_tools = [
            tool.get("type")
            for tool in request.tools
            if tool.get("type") in _OPENAI_BUILTIN_TOOL_TYPES
        ]
        if builtin_tools:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Built-in tools are not supported on this endpoint: {builtin_tools}",
            )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="OPENAI_API_KEY not configured",
        )

    client = OpenAIClient(api_key=api_key)
    input_messages = [msg.model_dump(exclude_none=True) for msg in request.input]
    result = await client.create_response(
        model=request.model,
        input=input_messages,
        temperature=request.temperature,
        max_output_tokens=request.max_output_tokens,
        tools=request.tools,
        tool_choice=request.tool_choice,
        instructions=request.instructions,
        **(request.model_extra or {}),
    )

    return models.GatewayOpenAIResponse(
        **result.model_dump(), cost=calculate_openai_cost(request.model, result)
    )


_OPENROUTER_WEB_SEARCH_MODEL_SUFFIX = ":online"


def _reject_openrouter_builtin_tools(request: models.OpenRouterInferenceRequest) -> None:
    if request.model.endswith(_OPENROUTER_WEB_SEARCH_MODEL_SUFFIX):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"The '{_OPENROUTER_WEB_SEARCH_MODEL_SUFFIX}' model suffix enables web search "
                f"and is not supported on this endpoint."
            ),
        )

    if (request.model_extra or {}).get("plugins"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "The 'plugins' field is not supported on this endpoint; "
                "only custom function tools are allowed."
            ),
        )

    if request.tools:
        builtin_tools = [
            tool.get("type") for tool in request.tools if tool.get("type") != "function"
        ]
        if builtin_tools:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Only custom function tools are supported on this endpoint. "
                    f"Disallowed built-in tools: {builtin_tools}"
                ),
            )


@gateway_router.post(
    "/openrouter/chat/completions/inference", response_model=models.GatewayOpenRouterCompletion
)
@cached_gateway_call
@handle_provider_errors("OpenRouter")
async def openrouter_chat_completion_inference(
    request: models.OpenRouterInferenceRequest,
) -> models.GatewayOpenRouterCompletion:
    _reject_openrouter_builtin_tools(request)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="OPENROUTER_API_KEY not configured",
        )

    client = OpenRouterClient(api_key=api_key)
    messages = [msg.model_dump() for msg in request.messages]
    result = await client.chat_completion(
        model=request.model,
        messages=messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        tools=request.tools,
        tool_choice=request.tool_choice,
        **(request.model_extra or {}),
    )

    return models.GatewayOpenRouterCompletion(
        **result.model_dump(), cost=calculate_openrouter_cost(result)
    )


@gateway_router.post(
    "/numinous-indicia/x-osint",
    response_model=models.GatewayNuminousIndiciaSignalsResponse,
)
@cached_gateway_call
@handle_provider_errors("NuminousIndicia")
async def numinous_indicia_x_osint(
    request: models.NuminousIndiciaXOsintRequest,
) -> models.GatewayNuminousIndiciaSignalsResponse:
    client = NuminousIndiciaClient()
    signals = await client.x_osint(account=request.account, limit=request.limit)

    return models.GatewayNuminousIndiciaSignalsResponse(
        signals=signals, cost=float(calculate_numinous_indicia_cost())
    )


@gateway_router.post(
    "/numinous-indicia/liveuamap",
    response_model=models.GatewayNuminousIndiciaSignalsResponse,
)
@cached_gateway_call
@handle_provider_errors("NuminousIndicia")
async def numinous_indicia_liveuamap(
    request: models.NuminousIndiciaLiveuamapRequest,
) -> models.GatewayNuminousIndiciaSignalsResponse:
    client = NuminousIndiciaClient()
    signals = await client.liveuamap(region=request.region, limit=request.limit)

    return models.GatewayNuminousIndiciaSignalsResponse(
        signals=signals, cost=float(calculate_numinous_indicia_cost())
    )


@gateway_router.post(
    "/lightning-rod/chat/completions", response_model=models.GatewayLightningRodCompletion
)
@cached_gateway_call
@handle_provider_errors("LightningRod")
async def lightning_rod_chat_completion(
    request: models.LightningRodInferenceRequest,
) -> models.GatewayLightningRodCompletion:
    api_key = os.getenv("LIGHTNING_ROD_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="LIGHTNING_ROD_API_KEY not configured",
        )

    client = LightningRodClient(api_key=api_key)
    messages = [msg.model_dump() for msg in request.messages]
    result = await client.chat_completion(
        model=request.model,
        messages=messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        **(request.model_extra or {}),
    )

    return models.GatewayLightningRodCompletion(
        **result.model_dump(), cost=calculate_lightning_rod_cost(result)
    )


@gateway_router.post(
    "/numinous-signals/causal-drivers/drivers",
    response_model=models.GatewayCausalDriversResponse,
)
@cached_gateway_call
@handle_provider_errors("NuminousSignals")
async def numinous_signals_causal_drivers(
    request: models.CausalDriversRequest,
) -> models.GatewayCausalDriversResponse:
    api_key = os.getenv("NUMINOUS_SIGNALS_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="NUMINOUS_SIGNALS_API_KEY not configured",
        )

    client = NuminousSignalsClient(api_key=api_key)
    result = await client.get_causal_drivers(
        event_id=request.event_id,
        topic=request.topic,
    )

    return models.GatewayCausalDriversResponse(
        **result.model_dump(), cost=float(calculate_causal_drivers_cost())
    )


@gateway_router.post(
    "/numinous-signals/deep-research/report",
    response_model=models.GatewayDeepResearchReportResponse,
)
@cached_gateway_call
@handle_provider_errors("NuminousSignals")
async def numinous_signals_deep_research(
    request: models.DeepResearchReportRequest,
) -> models.GatewayDeepResearchReportResponse:
    api_key = os.getenv("NUMINOUS_SIGNALS_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="NUMINOUS_SIGNALS_API_KEY not configured",
        )

    client = NuminousSignalsClient(api_key=api_key)
    result = await client.get_deep_research_report(
        event_id=request.event_id,
        polymarket_market_id=request.polymarket_market_id,
        title=request.title,
        topics=request.topics,
    )

    return models.GatewayDeepResearchReportResponse(
        **result.model_dump(), cost=float(calculate_deep_research_cost())
    )


@gateway_router.post(
    "/numinous-signals/corpus/search",
    response_model=models.GatewayCorpusSearchResponse,
)
@cached_gateway_call
@handle_provider_errors("NuminousSignals")
async def numinous_signals_corpus_search(
    request: models.CorpusSearchRequest,
) -> models.GatewayCorpusSearchResponse:
    api_key = os.getenv("NUMINOUS_SIGNALS_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="NUMINOUS_SIGNALS_API_KEY not configured",
        )

    client = NuminousSignalsClient(api_key=api_key)
    result = await client.search_corpus(
        query=request.query,
        max_results=request.max_results,
        published_after=request.published_after,
        published_before=request.published_before,
    )

    return models.GatewayCorpusSearchResponse(
        **result.model_dump(), cost=float(calculate_corpus_search_cost())
    )


@gateway_router.post(
    "/numinous-signals/corpus/fetch",
    response_model=models.GatewayCorpusFetchResponse,
)
@cached_gateway_call
@handle_provider_errors("NuminousSignals")
async def numinous_signals_corpus_fetch(
    request: models.CorpusFetchRequest,
) -> models.GatewayCorpusFetchResponse:
    api_key = os.getenv("NUMINOUS_SIGNALS_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="NUMINOUS_SIGNALS_API_KEY not configured",
        )

    client = NuminousSignalsClient(api_key=api_key)
    result = await client.fetch_corpus_source(request.source_id)

    return models.GatewayCorpusFetchResponse(
        **result.model_dump(), cost=float(calculate_corpus_fetch_cost())
    )


@gateway_router.post(
    "/numinous-signals/news",
    response_model=models.GatewayNewsFeedResponse,
)
@cached_gateway_call
@handle_provider_errors("NuminousSignals")
async def numinous_signals_news(
    request: models.NewsFeedRequest,
) -> models.GatewayNewsFeedResponse:
    api_key = os.getenv("NUMINOUS_SIGNALS_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="NUMINOUS_SIGNALS_API_KEY not configured",
        )

    client = NuminousSignalsClient(api_key=api_key)
    page = await client.get_news_feed(
        event_id=request.event_id,
        language=str(request.language) if request.language else None,
        min_impact_score=request.min_impact_score,
        order=request.order,
        published_within_hours=request.published_within_hours,
        limit=request.limit,
        offset=request.offset,
    )

    articles = [
        article
        for article in (NewsFeedArticle.from_item(item) for item in page.items)
        if article is not None
    ]

    return models.GatewayNewsFeedResponse(
        event_id=request.event_id,
        count=page.count,
        articles=articles,
        cost=float(calculate_news_feed_cost()),
    )


app.include_router(gateway_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
