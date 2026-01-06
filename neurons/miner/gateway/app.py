import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, status

from neurons.miner.gateway.cache import cached_gateway_call
from neurons.miner.gateway.error_handler import handle_provider_errors
from neurons.miner.gateway.providers.chutes import ChutesClient
from neurons.miner.gateway.providers.desearch import DesearchClient
from neurons.validator.models import numinous_client as models
from neurons.validator.models.chutes import ChuteStatus
from neurons.validator.models.chutes import calculate_cost as calculate_chutes_cost
from neurons.validator.models.desearch import DesearchEndpoint
from neurons.validator.models.desearch import calculate_cost as calculate_desearch_cost

logger = logging.getLogger(__name__)

env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    logger.warning(
        "No .env file found. Make sure to create a .env file in the gateway/ directory "
        "and add your API keys for Chutes and Desearch."
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


app.include_router(gateway_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
