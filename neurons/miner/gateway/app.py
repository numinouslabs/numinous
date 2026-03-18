import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, status

from neurons.miner.gateway.cache import cached_gateway_call
from neurons.miner.gateway.error_handler import handle_provider_errors
from neurons.miner.gateway.providers.chutes import ChutesClient
from neurons.miner.gateway.providers.desearch import DesearchClient
from neurons.miner.gateway.providers.numinous_indicia import NuminousIndiciaClient
from neurons.miner.gateway.providers.azure_openai import AzureOpenAIClient
from neurons.miner.gateway.providers.openai import OpenAIClient
from neurons.miner.gateway.providers.openrouter import OpenRouterClient
from neurons.miner.gateway.providers.perplexity import PerplexityClient
from neurons.miner.gateway.providers.vericore import VericoreClient
from neurons.validator.models import numinous_client as models
from neurons.validator.models.chutes import ChuteStatus
from neurons.validator.models.chutes import calculate_cost as calculate_chutes_cost
from neurons.validator.models.desearch import DesearchEndpoint
from neurons.validator.models.desearch import calculate_cost as calculate_desearch_cost
from neurons.validator.models.numinous_indicia import (
    calculate_cost as calculate_numinous_indicia_cost,
)
from neurons.validator.models.azure_openai import calculate_cost as calculate_azure_openai_cost
from neurons.validator.models.openai import calculate_cost as calculate_openai_cost
from neurons.validator.models.openrouter import calculate_cost as calculate_openrouter_cost
from neurons.validator.models.perplexity import calculate_cost as calculate_perplexity_cost
from neurons.validator.models.vericore import calculate_cost as calculate_vericore_cost

logger = logging.getLogger(__name__)

env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    logger.warning(
        "No .env file found. Make sure to create a .env file in the gateway/ directory "
        "and add your API keys for providers like Chutes, Desearch, OpenAI, Azure OpenAI, etc."
    )


app = FastAPI(title="Numinous API Gateway")
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


@gateway_router.post("/openai/responses", response_model=models.GatewayOpenAIResponse)
@cached_gateway_call
@handle_provider_errors("OpenAI")
async def openai_create_response(
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


@gateway_router.post("/azure-openai/responses", response_model=models.GatewayAzureOpenAIResponse)
@cached_gateway_call
@handle_provider_errors("AzureOpenAI")
async def azure_openai_create_response(
    request: models.AzureOpenAIInferenceRequest,
) -> models.GatewayAzureOpenAIResponse:
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    resource_name = os.getenv("AZURE_OPENAI_RESOURCE_NAME")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="AZURE_OPENAI_API_KEY not configured",
        )
    if not resource_name:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="AZURE_OPENAI_RESOURCE_NAME not configured",
        )

    client = AzureOpenAIClient(
        api_key=api_key,
        resource_name=resource_name,
        api_version=api_version,
    )
    input_messages = [msg.model_dump(exclude_none=True) for msg in request.input]
    result = await client.create_response(
        deployment=request.deployment,
        input=input_messages,
        temperature=request.temperature,
        max_output_tokens=request.max_output_tokens,
        tools=request.tools,
        tool_choice=request.tool_choice,
        instructions=request.instructions,
        **(request.model_extra or {}),
    )

    return models.GatewayAzureOpenAIResponse(
        **result.model_dump(), cost=calculate_azure_openai_cost(request.deployment, result)
    )


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


app.include_router(gateway_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)