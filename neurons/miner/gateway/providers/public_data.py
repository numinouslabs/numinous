import asyncio
import ipaddress
import logging
import os
import socket
from enum import StrEnum
from typing import Literal
from urllib.parse import parse_qs, urljoin, urlparse, urlunparse

import aiohttp
from pydantic import BaseModel

logger = logging.getLogger(__name__)

MAX_RESPONSE_SIZE_BYTES = 5 * 1024 * 1024  # 5MB
MAX_REDIRECTS = 3
REDIRECT_STATUSES = (301, 302, 303, 307, 308)
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


def _ensure_same_host_redirect(next_url: str, original_host: str) -> None:
    parsed = urlparse(next_url)
    if parsed.scheme not in ("http", "https"):
        raise UrlValidationError(f"Redirect to non-http(s) scheme blocked: {parsed.scheme}")
    next_host = (parsed.hostname or "").lower()
    if next_host != original_host:
        raise UrlValidationError(f"Cross-host redirect blocked: {original_host} -> {next_host}")


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


def _downgrade_method_after_redirect(
    status_code: int, current_method: str, current_body: bytes | None
) -> tuple[str, bytes | None]:
    if status_code == 303 or (status_code in (301, 302) and current_method == "POST"):
        return "GET", None
    return current_method, current_body


async def _fetch_with_redirects(
    session: aiohttp.ClientSession,
    initial_url: str,
    initial_method: str,
    initial_body: str | None,
    headers: dict[str, str],
    extra_params: dict[str, str],
    source: PublicDataSourceInfo,
    api_key: str | None,
    allowed_sources: list[PublicDataSourceInfo],
) -> tuple[int, dict[str, str], str]:
    original_host = (urlparse(initial_url).hostname or "").lower()
    current_url = initial_url
    current_method = initial_method.upper()
    current_body = initial_body.encode() if initial_body else None
    current_extra_params = extra_params

    for _ in range(MAX_REDIRECTS + 1):
        clean_url, merged_params = _extract_url_and_params(current_url, current_extra_params)
        async with session.request(
            method=current_method,
            url=clean_url,
            headers=headers,
            params=merged_params,
            data=current_body,
            allow_redirects=False,
        ) as response:
            response_status = response.status
            response_headers = dict(response.headers)
            response_url = str(response.url)
            location = response_headers.get("Location")
            is_terminal = response_status not in REDIRECT_STATUSES or not location
            if is_terminal:
                response_text = await response.text()
                return response_status, response_headers, response_text

        next_url = urljoin(response_url, location)
        _ensure_same_host_redirect(next_url, original_host)
        await validate_and_match_source(next_url, allowed_sources)

        current_method, current_body = _downgrade_method_after_redirect(
            response_status, current_method, current_body
        )
        current_url = next_url
        _, current_extra_params = _inject_auth(source, api_key, {}, {})

    raise UrlValidationError(f"Too many redirects (max {MAX_REDIRECTS})")


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

        request_timeout = aiohttp.ClientTimeout(total=min(timeout, 60.0))
        async with aiohttp.ClientSession(timeout=request_timeout) as session:
            response_status, response_headers, response_text = await _fetch_with_redirects(
                session=session,
                initial_url=url,
                initial_method=method,
                initial_body=body,
                headers=request_headers,
                extra_params=request_params,
                source=source,
                api_key=api_key,
                allowed_sources=self.__allowed_sources,
            )

        if len(response_text) > MAX_RESPONSE_SIZE_BYTES:
            logger.warning(
                "Response truncated to %d bytes for URL: %s",
                MAX_RESPONSE_SIZE_BYTES,
                url,
            )
            response_text = response_text[:MAX_RESPONSE_SIZE_BYTES]

        return PublicDataProxyResponse(
            status_code=response_status,
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
