"""Azure OpenAI gateway client (from azure_api_gateway_test.ipynb)."""

from __future__ import annotations

import os
from typing import Any, Callable

from openai import APIStatusError, AzureOpenAI, RateLimitError

# Defaults are non-secret; credentials must come from the environment / .env
_DEFAULT_ENDPOINT = "https://api.hku.hk"
_DEFAULT_API_VERSION = "2024-06-01"

API_KEY = (os.getenv("AZURE_OPENAI_API_KEY") or "").strip()
ENDPOINT = (os.getenv("AZURE_OPENAI_ENDPOINT") or _DEFAULT_ENDPOINT).strip()
API_VERSION = (os.getenv("AZURE_OPENAI_API_VERSION") or _DEFAULT_API_VERSION).strip()
DEPLOYMENT = (os.getenv("AZURE_OPENAI_DEPLOYMENT") or "gpt-5.1").strip()
REASONING_EFFORT = (os.getenv("AZURE_OPENAI_REASONING_EFFORT") or "medium").strip()


def get_client() -> AzureOpenAI:
    if not API_KEY:
        raise RuntimeError(
            "AZURE_OPENAI_API_KEY is not set. Copy .env.example to .env and add credentials, "
            "or set OPENAI_API_KEY and use coordinator_backend='openai'."
        )
    return AzureOpenAI(api_key=API_KEY, api_version=API_VERSION, azure_endpoint=ENDPOINT)


def chat_once(
    prompt: str,
    deployment: str | None = None,
    temperature: float | None = None,
    max_completion_tokens: int = 400,
    reasoning_effort: str | None = None,
    *,
    log: Callable[[str], None] | None = None,
) -> str:
    client = get_client()
    dep = (deployment or DEPLOYMENT).strip()
    kwargs: dict[str, Any] = {
        "model": dep,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_completion_tokens,
    }
    re = reasoning_effort if reasoning_effort is not None else REASONING_EFFORT
    if re:
        kwargs["reasoning_effort"] = re
    # GPT-5.x / o-series reasoning deployments only accept the default temperature.
    if temperature is not None and not re:
        kwargs["temperature"] = temperature
    try:
        resp = client.chat.completions.create(**kwargs)
    except RateLimitError as exc:
        raise RuntimeError(
            f"HKU gateway rate limit ({dep}) — wait and retry, or switch deployment. {exc}"
        ) from exc
    except APIStatusError as exc:
        if exc.status_code == 429:
            raise RuntimeError(
                f"HKU gateway quota/rate limit ({dep}, HTTP 429). {exc.message}"
            ) from exc
        raise
    message = resp.choices[0].message
    content = message.content or ""
    if log is not None:
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning is None and hasattr(message, "model_extra") and message.model_extra:
            reasoning = message.model_extra.get("reasoning_content")
        if reasoning:
            log(f"  [reasoning — {dep}]\n{reasoning}")
        usage = getattr(resp, "usage", None)
        if usage is not None:
            details = getattr(usage, "completion_tokens_details", None)
            reasoning_tokens = getattr(details, "reasoning_tokens", None) if details else None
            if reasoning_tokens:
                log(f"  [{dep}] reasoning tokens: {reasoning_tokens}")
        preview = content.strip().replace("\n", " ")
        if len(preview) > 400:
            preview = preview[:400] + "…"
        if preview:
            log(f"  [{dep}] response preview: {preview}")
        elif reasoning_tokens:
            log(
                f"  [{dep}] WARNING: empty response content "
                f"({reasoning_tokens} reasoning tokens — "
                "raise max_completion_tokens or lower reasoning effort)"
            )
        elif not content:
            log(f"  [{dep}] WARNING: empty response content")
    return content
