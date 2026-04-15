import asyncio
import ipaddress
import logging
import os
import socket
from enum import StrEnum
from typing import Literal
from urllib.parse import parse_qs, urlparse, urlunparse

import aiohttp
from pydantic import BaseModel

logger = logging.getLogger(__name__)

MAX_RESPONSE_SIZE_BYTES = 5 * 1024 * 1024  # 5MB
DEFAULT_SOURCES_URL = "https://numinous.earth/api/v3/miner/public-data/sources"
ALLOWED_RESPONSE_HEADERS = ("content-type", "date", "last-modified", "etag", "cache-control")


class AuthInjectionMethod(StrEnum):
    QUERY_PARAM = "QUERY_PARAM"
    HEADER = "HEADER"
    BEARER = "BEARER"


class PublicDataSourceInfo(BaseModel):
    name: str
    domain: str
    base_url: str | None = None
    category: str
    auth_injection_method: AuthInjectionMethod | None = None
    auth_param_name: str | None = None
    requires_auth: bool = False


class PublicDataProxyResponse(BaseModel):
    status_code: int
    response_headers: dict[str, str]
    response_body: str
    content_type: str | None = None
    source_name: str
    source_category: str


class UrlValidationError(Exception):
    pass


def _resolve_hostname_ips(hostname: str) -> list[str]:
    results = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    return [result[4][0] for result in results]


def _check_private_ip(ip_string: str) -> None:
    ip_address = ipaddress.ip_address(ip_string)
    if (
        ip_address.is_private
        or ip_address.is_loopback
        or ip_address.is_link_local
        or ip_address.is_reserved
    ):
        raise UrlValidationError(f"URL resolves to blocked IP range: {ip_string}")


def _find_matching_source(
    hostname: str, allowed_sources: list[PublicDataSourceInfo]
) -> PublicDataSourceInfo | None:
    for source in allowed_sources:
        if source.domain == hostname:
            return source
    return None


async def validate_and_match_source(
    url: str, allowed_sources: list[PublicDataSourceInfo]
) -> PublicDataSourceInfo:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UrlValidationError(f"Only HTTP and HTTPS schemes are allowed, got: {parsed.scheme}")

    hostname = parsed.hostname
    if not hostname:
        raise UrlValidationError("No hostname found in URL")

    hostname = hostname.lower()
    source = _find_matching_source(hostname, allowed_sources)
    if source is None:
        raise UrlValidationError(f"Domain not in whitelist: {hostname}")

    if source.base_url and not url.startswith(source.base_url):
        raise UrlValidationError(
            f"URL does not match allowed base URL for {source.name}. "
            f"Expected prefix: {source.base_url}"
        )

    resolved_ips = await asyncio.to_thread(_resolve_hostname_ips, hostname)
    for resolved_ip in resolved_ips:
        _check_private_ip(resolved_ip)

    return source


def _inject_auth(
    source: PublicDataSourceInfo,
    api_key: str | None,
    headers: dict[str, str],
    query_params: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    if not api_key or not source.auth_injection_method:
        return headers, query_params

    headers = dict(headers)
    query_params = dict(query_params)

    if source.auth_injection_method == AuthInjectionMethod.QUERY_PARAM:
        query_params[source.auth_param_name] = api_key
    elif source.auth_injection_method == AuthInjectionMethod.HEADER:
        headers[source.auth_param_name] = api_key
    elif source.auth_injection_method == AuthInjectionMethod.BEARER:
        headers["Authorization"] = f"Bearer {api_key}"

    return headers, query_params


def _extract_url_and_params(url: str, extra_params: dict[str, str]) -> tuple[str, dict[str, str]]:
    parsed_url = urlparse(url)
    existing_params = parse_qs(parsed_url.query, keep_blank_values=True)
    merged_params = {key: values[0] for key, values in existing_params.items()}
    merged_params.update(extra_params)
    clean_url = urlunparse(parsed_url._replace(query=""))
    return clean_url, merged_params


def _filter_response_headers(raw_headers: dict[str, str]) -> dict[str, str]:
    return {
        key: value for key, value in raw_headers.items() if key.lower() in ALLOWED_RESPONSE_HEADERS
    }


def _get_env_var_name(source: PublicDataSourceInfo) -> str:
    return f"{source.name.upper()}_API_KEY"


def _get_api_key_for_source(source: PublicDataSourceInfo) -> str | None:
    if not source.requires_auth:
        return None
    return os.getenv(_get_env_var_name(source))


class PublicDataProxyClient:
    def __init__(self, allowed_sources: list[PublicDataSourceInfo]) -> None:
        self.__allowed_sources = allowed_sources

    @property
    def allowed_sources(self) -> list[PublicDataSourceInfo]:
        return self.__allowed_sources

    async def proxy_request(
        self,
        url: str,
        method: Literal["GET", "POST", "PUT", "DELETE"] = "GET",
        headers: dict[str, str] | None = None,
        query_params: dict[str, str] | None = None,
        body: str | None = None,
        timeout: float = 30.0,
    ) -> PublicDataProxyResponse:
        request_headers = dict(headers or {})
        request_params = dict(query_params or {})

        source = await validate_and_match_source(url, self.__allowed_sources)

        api_key = _get_api_key_for_source(source)

        request_headers, request_params = _inject_auth(
            source, api_key, request_headers, request_params
        )

        clean_url, merged_params = _extract_url_and_params(url, request_params)

        request_timeout = aiohttp.ClientTimeout(total=min(timeout, 60.0))
        async with aiohttp.ClientSession(timeout=request_timeout) as session:
            async with session.request(
                method=method.upper(),
                url=clean_url,
                headers=request_headers,
                params=merged_params,
                data=body.encode() if body else None,
                allow_redirects=False,
            ) as response:
                response_text = await response.text()
                response_headers = dict(response.headers)

        if len(response_text) > MAX_RESPONSE_SIZE_BYTES:
            logger.warning(
                "Response truncated to %d bytes for URL: %s",
                MAX_RESPONSE_SIZE_BYTES,
                url,
            )
            response_text = response_text[:MAX_RESPONSE_SIZE_BYTES]

        return PublicDataProxyResponse(
            status_code=response.status,
            response_headers=_filter_response_headers(response_headers),
            response_body=response_text,
            content_type=response_headers.get("Content-Type"),
            source_name=source.name,
            source_category=source.category,
        )


def fetch_sources_from_prod() -> list[PublicDataSourceInfo]:
    import httpx

    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.get(DEFAULT_SOURCES_URL)

        if response.status_code != 200:
            logger.warning(
                "Failed to fetch public data sources from prod: HTTP %d", response.status_code
            )
            return []

        raw_sources = response.json().get("sources", [])
        sources = []
        for raw in raw_sources:
            sources.append(
                PublicDataSourceInfo(
                    name=raw["name"],
                    domain=raw["domain"],
                    base_url=raw.get("base_url"),
                    category=raw["category"],
                    requires_auth=raw.get("requires_auth", False),
                )
            )
        return sources
    except Exception as exc:
        logger.warning("Failed to fetch public data sources from prod: %s", exc)
        return []


def log_source_warnings(sources: list[PublicDataSourceInfo]) -> None:
    auth_sources = [s for s in sources if s.requires_auth]
    if not auth_sources:
        return

    missing = [s for s in auth_sources if not os.getenv(_get_env_var_name(s))]
    if missing:
        names = ", ".join(f"{s.name} ({_get_env_var_name(s)})" for s in missing)
        logger.warning(
            "Public data sources requiring API keys not set in .env: %s. "
            "Requests to these sources will fail until configured.",
            names,
        )
