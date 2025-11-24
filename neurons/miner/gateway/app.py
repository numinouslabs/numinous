import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, status

from neurons.miner.gateway.cache import cached_gateway_call
from neurons.miner.gateway.error_handler import handle_provider_errors
from neurons.miner.gateway.providers.chutes import ChutesClient
from neurons.miner.gateway.providers.desearch import DesearchClient
from neurons.validator.models.chutes import ChutesCompletion, ChuteStatus
from neurons.validator.models.desearch import (
    AISearchResponse,
    WebCrawlResponse,
    WebLinksResponse,
    WebSearchResponse,
)
from neurons.validator.models.numinous_client import (
    ChutesInferenceRequest,
    DesearchAISearchRequest,
    DesearchWebCrawlRequest,
    DesearchWebLinksRequest,
    DesearchWebSearchRequest,
)

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


@gateway_router.post("/chutes/chat/completions", response_model=ChutesCompletion)
@cached_gateway_call
@handle_provider_errors("Chutes")
async def chutes_chat_completion(request: ChutesInferenceRequest) -> ChutesCompletion:
    api_key = os.getenv("CHUTES_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="CHUTES_API_KEY not configured",
        )

    client = ChutesClient(api_key=api_key)
    messages = [msg.model_dump() for msg in request.messages]
    return await client.chat_completion(
        model=request.model,
        messages=messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        tools=request.tools,
        tool_choice=request.tool_choice,
        **(request.model_extra or {}),
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


@gateway_router.post("/desearch/ai/search", response_model=AISearchResponse)
@cached_gateway_call
@handle_provider_errors("Desearch")
async def desearch_ai_search(request: DesearchAISearchRequest) -> AISearchResponse:
    api_key = os.getenv("DESEARCH_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="DESEARCH_API_KEY not configured",
        )

    client = DesearchClient(api_key=api_key)
    return await client.ai_search(
        prompt=request.prompt,
        model=request.model,
        tools=request.tools,
        date_filter=request.date_filter,
        result_type=request.result_type,
        system_message=request.system_message,
        count=request.count,
    )


@gateway_router.post("/desearch/ai/links", response_model=WebLinksResponse)
@cached_gateway_call
@handle_provider_errors("Desearch")
async def desearch_web_links_search(
    request: DesearchWebLinksRequest,
) -> WebLinksResponse:
    api_key = os.getenv("DESEARCH_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="DESEARCH_API_KEY not configured",
        )

    client = DesearchClient(api_key=api_key)
    return await client.web_links_search(
        prompt=request.prompt, model=request.model, tools=request.tools, count=request.count
    )


@gateway_router.post("/desearch/web/search", response_model=WebSearchResponse)
@cached_gateway_call
@handle_provider_errors("Desearch")
async def desearch_web_search(request: DesearchWebSearchRequest) -> WebSearchResponse:
    api_key = os.getenv("DESEARCH_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="DESEARCH_API_KEY not configured",
        )

    client = DesearchClient(api_key=api_key)
    return await client.web_search(
        query=request.query, num_results=request.num, start=request.start
    )


@gateway_router.post("/desearch/web/crawl", response_model=WebCrawlResponse)
@cached_gateway_call
@handle_provider_errors("Desearch")
async def desearch_web_crawl(request: DesearchWebCrawlRequest) -> WebCrawlResponse:
    api_key = os.getenv("DESEARCH_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="DESEARCH_API_KEY not configured",
        )

    client = DesearchClient(api_key=api_key)
    return await client.web_crawl(url=request.url)


app.include_router(gateway_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
