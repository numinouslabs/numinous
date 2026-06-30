TRACK_ALLOWED_PREFIXES: dict[str, list[str]] = {
    "MAIN": ["*"],
    "SIGNAL": [
        "/api/gateway/numinous-indicia/",
        "/api/gateway/lightning-rod/",
        "/api/gateway/numinous-signals/",
        "/api/gateway/openai/responses/inference",
        "/api/gateway/openrouter/chat/completions/inference",
    ],
}
