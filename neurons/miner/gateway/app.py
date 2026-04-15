import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, status

from neurons.miner.gateway.cache import cached_gateway_call
from neurons.miner.gateway.error_handler import handle_provider_errors
from neurons.miner.gateway.providers.chutes import ChutesClient
from neurons.miner.gateway.providers.desearch import DesearchClient
from neurons.miner.gateway.providers.lightning_rod import LightningRodClient
from neurons.miner.gateway.providers.lunar_crush import LunarCrushClient
from neurons.miner.gateway.providers.numinous_indicia import NuminousIndiciaClient
from neurons.miner.gateway.providers.numinous_signals import NuminousSignalsClient
from neurons.miner.gateway.providers.openai import OpenAIClient
from neurons.miner.gateway.providers.openrouter import OpenRouterClient
from neurons.miner.gateway.providers.perplexity import PerplexityClient
from neurons.miner.gateway.providers.public_data import (
    PublicDataProxyClient,
    PublicDataSourceInfo,
    UrlValidationError,
    fetch_sources_from_prod,
    log_source_warnings,
)
from neurons.miner.gateway.providers.unusual_whales import UnusualWhalesClient
from neurons.miner.gateway.providers.vericore import VericoreClient
from neurons.validator.models import numinous_client as models
from neurons.validator.models.chutes import ChuteStatus
from neurons.validator.models.chutes import calculate_cost as calculate_chutes_cost
from neurons.validator.models.desearch import DesearchEndpoint
from neurons.validator.models.desearch import calculate_cost as calculate_desearch_cost
from neurons.validator.models.lightning_rod import calculate_cost as calculate_lightning_rod_cost
from neurons.validator.models.lunar_crush import calculate_cost as calculate_lunar_crush_cost
from neurons.validator.models.numinous_indicia import (
    calculate_cost as calculate_numinous_indicia_cost,
)
from neurons.validator.models.numinous_signals import calculate_causal_drivers_cost
from neurons.validator.models.numinous_signals import (
    calculate_cost as calculate_numinous_signals_cost,
)
from neurons.validator.models.numinous_signals import calculate_deep_research_cost
from neurons.validator.models.openai import calculate_cost as calculate_openai_cost
from neurons.validator.models.openrouter import calculate_cost as calculate_openrouter_cost
from neurons.validator.models.perplexity import calculate_cost as calculate_perplexity_cost
from neurons.validator.models.unusual_whales import calculate_cost as calculate_unusual_whales_cost
from neurons.validator.models.vericore import calculate_cost as calculate_vericore_cost

logger = logging.getLogger(__name__)

env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    logger.warning(
        "No .env file found. Make sure to create a .env file in the gateway/ directory "
        "and add your API keys for Chutes and Desearch."
    )


_public_data_sources: list[PublicDataSourceInfo] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _public_data_sources
    logger.info("Fetching public data sources...")
    _public_data_sources = fetch_sources_from_prod()
    logger.info("Loaded %d public data source(s)", len(_public_data_sources))
    log_source_warnings(_public_data_sources)
    yield


app = FastAPI(title="Numinous API Gateway", lifespan=lifespan)
gateway_router = APIRouter(prefix="/api/gateway")


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "API Gateway"}


@gateway_router.post("/chutes/chat/completions", response_model=models.GatewayChutesCompletion)
@cached_gateway_call
@handle_provider_errors("Chutes")
async def chutes_chat_completion(request: models.ChutesInferenceRequest) -> models.ChutesCompletion:
    api_key = os.getenv("CHUTES_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="CHUTES_API_KEY not configured",
        )

    client = ChutesClient(api_key=api_key)
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

    return models.GatewayChutesCompletion(
        **result.model_dump(), cost=calculate_chutes_cost(request.model, result)
    )


@gateway_router.get("/chutes/status", response_model=list[ChuteStatus])
@handle_provider_errors("Chutes")
async def get_chutes_status() -> list[ChuteStatus]:
    api_key = os.getenv("CHUTES_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="CHUTES_API_KEY not configured",
        )

    client = ChutesClient(api_key=api_key)
    return await client.get_chutes_status()


@gateway_router.post("/desearch/ai/search", response_model=models.GatewayDesearchAISearchResponse)
@cached_gateway_call
@handle_provider_errors("Desearch")
async def desearch_ai_search(
    request: models.DesearchAISearchRequest,
) -> models.GatewayDesearchAISearchResponse:
    api_key = os.getenv("DESEARCH_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="DESEARCH_API_KEY not configured",
        )

    client = DesearchClient(api_key=api_key)
    result = await client.ai_search(
        prompt=request.prompt,
        model=request.model,
        tools=request.tools,
        date_filter=request.date_filter,
        result_type=request.result_type,
        system_message=request.system_message,
        count=request.count,
    )

    return models.GatewayDesearchAISearchResponse(
        **result.model_dump(),
        cost=calculate_desearch_cost(DesearchEndpoint.AI_SEARCH, request.model),
    )


@gateway_router.post("/desearch/ai/links", response_model=models.GatewayDesearchWebLinksResponse)
@cached_gateway_call
@handle_provider_errors("Desearch")
async def desearch_web_links_search(
    request: models.DesearchWebLinksRequest,
) -> models.GatewayDesearchWebLinksResponse:
    api_key = os.getenv("DESEARCH_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="DESEARCH_API_KEY not configured",
        )

    client = DesearchClient(api_key=api_key)
    result = await client.web_links_search(
        prompt=request.prompt, model=request.model, tools=request.tools, count=request.count
    )
    return models.GatewayDesearchWebLinksResponse(
        **result.model_dump(),
        cost=calculate_desearch_cost(DesearchEndpoint.AI_WEB_SEARCH, request.model),
    )


@gateway_router.post("/desearch/web/search", response_model=models.GatewayDesearchWebSearchResponse)
@cached_gateway_call
@handle_provider_errors("Desearch")
async def desearch_web_search(
    request: models.DesearchWebSearchRequest,
) -> models.GatewayDesearchWebSearchResponse:
    api_key = os.getenv("DESEARCH_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="DESEARCH_API_KEY not configured",
        )

    client = DesearchClient(api_key=api_key)
    result = await client.web_search(
        query=request.query, num_results=request.num, start=request.start
    )
    return models.GatewayDesearchWebSearchResponse(
        **result.model_dump(),
        cost=calculate_desearch_cost(DesearchEndpoint.WEB_SEARCH),
    )


@gateway_router.post("/desearch/web/crawl", response_model=models.GatewayDesearchWebCrawlResponse)
@cached_gateway_call
@handle_provider_errors("Desearch")
async def desearch_web_crawl(
    request: models.DesearchWebCrawlRequest,
) -> models.GatewayDesearchWebCrawlResponse:
    api_key = os.getenv("DESEARCH_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="DESEARCH_API_KEY not configured",
        )

    client = DesearchClient(api_key=api_key)
    result = await client.web_crawl(url=request.url)

    return models.GatewayDesearchWebCrawlResponse(
        **result.model_dump(),
        cost=calculate_desearch_cost(DesearchEndpoint.WEB_CRAWL),
    )


@gateway_router.post("/desearch/x/search", response_model=models.GatewayDesearchXSearchResponse)
@cached_gateway_call
@handle_provider_errors("Desearch")
async def desearch_x_search(
    request: models.DesearchXSearchRequest,
) -> models.GatewayDesearchXSearchResponse:
    api_key = os.getenv("DESEARCH_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="DESEARCH_API_KEY not configured",
        )

    client = DesearchClient(api_key=api_key)
    result = await client.x_search(
        query=request.query,
        sort=request.sort,
        user=request.user,
        start_date=request.start_date,
        end_date=request.end_date,
        lang=request.lang,
        verified=request.verified,
        blue_verified=request.blue_verified,
        is_quote=request.is_quote,
        is_video=request.is_video,
        is_image=request.is_image,
        min_retweets=request.min_retweets,
        min_replies=request.min_replies,
        min_likes=request.min_likes,
        count=request.count,
    )

    return models.GatewayDesearchXSearchResponse(
        posts=result,
        cost=calculate_desearch_cost(DesearchEndpoint.X_SEARCH),
    )


@gateway_router.post("/desearch/x/post", response_model=models.GatewayDesearchXPostResponse)
@cached_gateway_call
@handle_provider_errors("Desearch")
async def desearch_x_post(
    request: models.DesearchXPostRequest,
) -> models.GatewayDesearchXPostResponse:
    api_key = os.getenv("DESEARCH_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="DESEARCH_API_KEY not configured",
        )

    client = DesearchClient(api_key=api_key)
    result = await client.fetch_x_post(post_id=request.post_id)

    return models.GatewayDesearchXPostResponse(
        **result.model_dump(),
        cost=calculate_desearch_cost(DesearchEndpoint.FETCH_X_POST),
    )


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


async def _run_openai_request(
    request: models.OpenAIInferenceRequest,
) -> models.GatewayOpenAIResponse:
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


@gateway_router.post("/openai/responses", response_model=models.GatewayOpenAIResponse)
@cached_gateway_call
@handle_provider_errors("OpenAI")
async def openai_create_response(
    request: models.OpenAIInferenceRequest,
) -> models.GatewayOpenAIResponse:
    if request.tools:
        disallowed = [
            tool.get("type")
            for tool in request.tools
            if tool.get("type") in _OPENAI_BUILTIN_TOOL_TYPES - {"web_search", "web_search_preview"}
        ]
        if disallowed:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Only web_search is supported as a built-in tool on this endpoint. "
                    f"Disallowed built-in tools: {disallowed}"
                ),
            )

    return await _run_openai_request(request)


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

    return await _run_openai_request(request)


@gateway_router.post("/perplexity/chat/completions")
@cached_gateway_call
@handle_provider_errors("Perplexity")
async def perplexity_chat_completion(request: models.PerplexityInferenceRequest):
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="PERPLEXITY_API_KEY not configured",
        )

    client = PerplexityClient(api_key=api_key)
    messages = [msg.model_dump() for msg in request.messages]
    result = await client.chat_completion(
        model=request.model,
        messages=messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        search_recency_filter=request.search_recency_filter,
        **(request.model_extra or {}),
    )

    return models.GatewayPerplexityCompletion(
        **result.model_dump(), cost=calculate_perplexity_cost(request.model, result)
    )


@gateway_router.post("/vericore/calculate-rating", response_model=models.GatewayVericoreResponse)
@cached_gateway_call
@handle_provider_errors("Vericore")
async def vericore_calculate_rating(
    request: models.VericoreCalculateRatingRequest,
) -> models.GatewayVericoreResponse:
    api_key = os.getenv("VERICORE_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="VERICORE_API_KEY not configured",
        )

    client = VericoreClient(api_key=api_key)
    result = await client.calculate_rating(
        statement=request.statement,
        generate_preview=request.generate_preview,
    )

    return models.GatewayVericoreResponse(**result.model_dump(), cost=calculate_vericore_cost())


@gateway_router.post(
    "/openrouter/chat/completions", response_model=models.GatewayOpenRouterCompletion
)
@cached_gateway_call
@handle_provider_errors("OpenRouter")
async def openrouter_chat_completion(
    request: models.OpenRouterInferenceRequest,
) -> models.GatewayOpenRouterCompletion:
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


@gateway_router.post("/lunar-crush/topic", response_model=models.GatewayLunarCrushTopicResponse)
@cached_gateway_call
@handle_provider_errors("LunarCrush")
async def lunar_crush_get_topic(
    request: models.LunarCrushTopicRequest,
) -> models.GatewayLunarCrushTopicResponse:
    api_key = os.getenv("LUNAR_CRUSH_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="LUNAR_CRUSH_API_KEY not configured",
        )
    client = LunarCrushClient(api_key=api_key)
    result = await client.get_topic(topic=request.topic)
    return models.GatewayLunarCrushTopicResponse(
        **result.model_dump(), cost=calculate_lunar_crush_cost()
    )


@gateway_router.post(
    "/lunar-crush/time-series", response_model=models.GatewayLunarCrushTimeSeriesResponse
)
@cached_gateway_call
@handle_provider_errors("LunarCrush")
async def lunar_crush_get_time_series(
    request: models.LunarCrushTimeSeriesRequest,
) -> models.GatewayLunarCrushTimeSeriesResponse:
    api_key = os.getenv("LUNAR_CRUSH_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="LUNAR_CRUSH_API_KEY not configured",
        )
    client = LunarCrushClient(api_key=api_key)
    result = await client.get_topic_time_series(topic=request.topic, bucket=request.bucket)
    return models.GatewayLunarCrushTimeSeriesResponse(
        **result.model_dump(), cost=calculate_lunar_crush_cost()
    )


@gateway_router.post("/lunar-crush/news", response_model=models.GatewayLunarCrushNewsResponse)
@cached_gateway_call
@handle_provider_errors("LunarCrush")
async def lunar_crush_get_news(
    request: models.LunarCrushNewsRequest,
) -> models.GatewayLunarCrushNewsResponse:
    api_key = os.getenv("LUNAR_CRUSH_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="LUNAR_CRUSH_API_KEY not configured",
        )
    client = LunarCrushClient(api_key=api_key)
    result = await client.get_topic_news(topic=request.topic)
    return models.GatewayLunarCrushNewsResponse(
        **result.model_dump(), cost=calculate_lunar_crush_cost()
    )


@gateway_router.post("/lunar-crush/whatsup", response_model=models.GatewayLunarCrushWhatsupResponse)
@cached_gateway_call
@handle_provider_errors("LunarCrush")
async def lunar_crush_get_whatsup(
    request: models.LunarCrushWhatsupRequest,
) -> models.GatewayLunarCrushWhatsupResponse:
    api_key = os.getenv("LUNAR_CRUSH_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="LUNAR_CRUSH_API_KEY not configured",
        )
    client = LunarCrushClient(api_key=api_key)
    result = await client.get_topic_whatsup(topic=request.topic)
    return models.GatewayLunarCrushWhatsupResponse(
        **result.model_dump(), cost=calculate_lunar_crush_cost()
    )


@gateway_router.post("/lunar-crush/posts", response_model=models.GatewayLunarCrushPostsResponse)
@cached_gateway_call
@handle_provider_errors("LunarCrush")
async def lunar_crush_get_posts(
    request: models.LunarCrushPostsRequest,
) -> models.GatewayLunarCrushPostsResponse:
    api_key = os.getenv("LUNAR_CRUSH_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="LUNAR_CRUSH_API_KEY not configured",
        )
    client = LunarCrushClient(api_key=api_key)
    result = await client.get_topic_posts(topic=request.topic, start=request.start, end=request.end)
    return models.GatewayLunarCrushPostsResponse(
        **result.model_dump(), cost=calculate_lunar_crush_cost()
    )


@gateway_router.post(
    "/lunar-crush/coins-list", response_model=models.GatewayLunarCrushCoinsListResponse
)
@cached_gateway_call
@handle_provider_errors("LunarCrush")
async def lunar_crush_get_coins_list(
    request: models.LunarCrushCoinsListRequest,
) -> models.GatewayLunarCrushCoinsListResponse:
    api_key = os.getenv("LUNAR_CRUSH_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="LUNAR_CRUSH_API_KEY not configured",
        )
    client = LunarCrushClient(api_key=api_key)
    result = await client.get_coins_list()
    return models.GatewayLunarCrushCoinsListResponse(
        **result.model_dump(), cost=calculate_lunar_crush_cost()
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
    "/numinous-signals/signals",
    response_model=models.GatewayNuminousSignalsResponse,
)
@cached_gateway_call
@handle_provider_errors("NuminousSignals")
async def numinous_signals_compute(
    request: models.NuminousSignalsRequest,
) -> models.GatewayNuminousSignalsResponse:
    api_key = os.getenv("NUMINOUS_SIGNALS_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="NUMINOUS_SIGNALS_API_KEY not configured",
        )

    client = NuminousSignalsClient(api_key=api_key)
    result = await client.compute_signals(
        market=request.market,
        question=request.question,
        relevance_threshold=request.relevance_threshold,
        max_events_per_source=request.max_events_per_source,
        time_window_hours=request.time_window_hours,
    )

    return models.GatewayNuminousSignalsResponse(
        **result.model_dump(), cost=calculate_numinous_signals_cost()
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
    "/unusual-whales/news/headlines",
    response_model=models.GatewayUnusualWhalesHeadlinesResponse,
)
@cached_gateway_call
@handle_provider_errors("UnusualWhales")
async def unusual_whales_get_news_headlines(
    request: models.UnusualWhalesHeadlinesRequest,
) -> models.GatewayUnusualWhalesHeadlinesResponse:
    api_key = os.getenv("UNUSUAL_WHALES_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="UNUSUAL_WHALES_API_KEY not configured",
        )

    client = UnusualWhalesClient(api_key=api_key)
    result = await client.get_news_headlines(
        sources=request.sources,
        search_term=request.search_term,
        ticker=request.ticker,
        major_only=request.major_only,
        limit=request.limit,
        page=request.page,
    )

    return models.GatewayUnusualWhalesHeadlinesResponse(
        **result.model_dump(), cost=calculate_unusual_whales_cost()
    )


@gateway_router.post(
    "/public-data/fetch",
    response_model=models.GatewayPublicDataProxyResponse,
)
@cached_gateway_call
@handle_provider_errors("PublicData")
async def public_data_proxy(
    request: models.PublicDataProxyRequest,
) -> models.GatewayPublicDataProxyResponse:
    client = PublicDataProxyClient(allowed_sources=_public_data_sources)

    try:
        result = await client.proxy_request(
            url=request.url,
            method=request.method,
            headers=request.headers,
            query_params=request.query_params,
            body=request.body,
            timeout=request.timeout,
        )
    except UrlValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    return models.GatewayPublicDataProxyResponse(
        status_code=result.status_code,
        response_headers=result.response_headers,
        response_body=result.response_body,
        content_type=result.content_type,
        source_name=result.source_name,
        source_category=result.source_category,
        cost=0.0,
    )


app.include_router(gateway_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
