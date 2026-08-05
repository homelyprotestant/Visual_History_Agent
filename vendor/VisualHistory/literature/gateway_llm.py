"""Unified cloud routing for Azure, Gemini, and direct OpenAI."""

from __future__ import annotations

import os
import threading
from typing import Callable, Literal

from openai import APIStatusError, OpenAI, RateLimitError

from VisualHistory.gemini_llm import default_gemini_model, gemini_available, gemini_chat

GatewayBackend = Literal["azure", "gemini", "openai", "azure-openai-fallback"]
ActiveGatewayBackend = Literal["azure", "gemini", "openai"]

DEFAULT_AZURE_GATEWAY_DEPLOYMENT = "gpt-5.5"
DEFAULT_GEMINI_GATEWAY_DEPLOYMENT = "gemini-3.1-pro-preview"
DEFAULT_OPENAI_MODEL = "gpt-5.1"
DEFAULT_AZURE_REASONING_EFFORT = "medium"
DEFAULT_SYNTHESIS_REASONING_EFFORT = "low"
# Reasoning models count thinking + output toward max_completion_tokens.
REASONING_OUTPUT_BUFFER = 8192
_OPENAI_FAILOVER_ACTIVE = threading.Event()


def _normalize_gateway_backend(raw: str | None) -> GatewayBackend:
    value = (raw or "azure").strip().lower()
    if value in {"gemini", "google"}:
        return "gemini"
    if value in {"openai", "direct_openai", "direct-openai"}:
        return "openai"
    if value in {
        "azure-openai-fallback",
        "azure_openai_fallback",
        "azure+openai",
        "fallback",
        "auto-fallback",
    }:
        return "azure-openai-fallback"
    return "azure"


def gateway_backend() -> GatewayBackend:
    """Cloud LLM family for orchestrator, Finalize, and cloud RCS."""
    raw = (
        os.getenv("LITERATURE_GATEWAY_BACKEND")
        or os.getenv("LITERATURE_CLOUD_LLM")
        or os.getenv("LITERATURE_ORCHESTRATOR_BACKEND")
        or os.getenv("LITERATURE_SYNTHESIS_BACKEND")
        or "azure"
    )
    return _normalize_gateway_backend(raw)


def _azure_available() -> bool:
    try:
        from VisualHistory.llm import API_KEY, ENDPOINT

        return bool(API_KEY and ENDPOINT)
    except Exception:
        return False


def _openai_available() -> bool:
    return bool((os.getenv("OPENAI_API_KEY") or "").strip())


def active_gateway_backend() -> ActiveGatewayBackend:
    configured = gateway_backend()
    if configured == "azure-openai-fallback":
        if _OPENAI_FAILOVER_ACTIVE.is_set() or not _azure_available():
            return "openai"
        return "azure"
    return configured


def activate_openai_fallback() -> bool:
    """Activate sticky OpenAI failover; return whether failover was configured."""
    if gateway_backend() != "azure-openai-fallback":
        return False
    if not _openai_available():
        raise RuntimeError(
            "Azure failed and OPENAI_API_KEY is unavailable for configured fallback"
        )
    _OPENAI_FAILOVER_ACTIVE.set()
    return True


def gateway_available() -> bool:
    active = active_gateway_backend()
    if active == "openai":
        return _openai_available()
    if active == "gemini":
        return _azure_available()
    return _azure_available()


def gateway_azure_deployment() -> str:
    return (
        os.getenv("LITERATURE_GATEWAY_DEPLOYMENT")
        or os.getenv("LITERATURE_AZURE_DEPLOYMENT")
        or os.getenv("LITERATURE_ORCHESTRATOR_DEPLOYMENT")
        or os.getenv("LITERATURE_ASK_ANSWER_DEPLOYMENT")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        or DEFAULT_AZURE_GATEWAY_DEPLOYMENT
    ).strip()


def gateway_gemini_deployment() -> str:
    return (
        os.getenv("LITERATURE_GATEWAY_DEPLOYMENT")
        or os.getenv("AZURE_GEMINI_DEPLOYMENT")
        or os.getenv("GEMINI_DEPLOYMENT")
        or os.getenv("GEMINI_MODEL")
        or DEFAULT_GEMINI_GATEWAY_DEPLOYMENT
    ).strip()


def gateway_openai_model() -> str:
    return (
        os.getenv("LITERATURE_GATEWAY_DEPLOYMENT")
        or os.getenv("LITERATURE_OPENAI_MODEL")
        or os.getenv("OPENAI_MODEL")
        or DEFAULT_OPENAI_MODEL
    ).strip()


def gateway_deployment() -> str:
    """Single deployment name for the active gateway backend."""
    active = active_gateway_backend()
    if active == "gemini":
        return gateway_gemini_deployment()
    if active == "openai":
        return gateway_openai_model()
    return gateway_azure_deployment()


def gateway_label() -> str:
    configured = gateway_backend()
    active = active_gateway_backend()
    if configured == "azure-openai-fallback":
        return (
            f"azure/{gateway_azure_deployment()}→"
            f"openai/{gateway_openai_model()} fallback"
        )
    if active == "gemini":
        from VisualHistory.gemini_llm import gemini_model_label

        return f"gemini/{gemini_model_label(gateway_deployment())}"
    if active == "openai":
        return f"openai/{gateway_deployment()}"
    return f"{active}/{gateway_deployment()}"


def _base_gateway_reasoning_effort() -> str:
    return (
        os.getenv("LITERATURE_GATEWAY_REASONING_EFFORT")
        or os.getenv("LITERATURE_RCS_REASONING_EFFORT")
        or os.getenv("AZURE_OPENAI_REASONING_EFFORT")
        or DEFAULT_AZURE_REASONING_EFFORT
    ).strip()


def gateway_reasoning_effort(*, large_completion: bool = False) -> str:
    if large_completion:
        return (
            os.getenv("LITERATURE_SYNTHESIS_REASONING_EFFORT")
            or os.getenv("LITERATURE_GATEWAY_REASONING_EFFORT")
            or DEFAULT_SYNTHESIS_REASONING_EFFORT
        ).strip()
    return _base_gateway_reasoning_effort()


def _openai_chat(
    prompt: str,
    *,
    model: str,
    max_tokens: int,
    temperature: float | None,
    reasoning_effort: str | None,
    log: Callable[[str], None] | None,
) -> str:
    client = OpenAI(api_key=(os.getenv("OPENAI_API_KEY") or "").strip())
    re = (
        reasoning_effort
        if reasoning_effort is not None
        else gateway_reasoning_effort(large_completion=max_tokens >= 8192)
    )
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens + (REASONING_OUTPUT_BUFFER if re else 0),
    }
    if re:
        kwargs["reasoning_effort"] = re
    elif temperature is not None:
        kwargs["temperature"] = temperature
    try:
        response = client.chat.completions.create(**kwargs)
    except RateLimitError as exc:
        raise RuntimeError(f"OpenAI rate limit ({model}). {exc}") from exc
    except APIStatusError as exc:
        if exc.status_code == 429:
            raise RuntimeError(f"OpenAI quota/rate limit ({model}). {exc.message}") from exc
        raise
    text = (response.choices[0].message.content or "").strip()
    if log and text:
        preview = text.replace("\n", " ")
        log(f"  [{model}] response preview: {preview[:400]}{'…' if len(preview) > 400 else ''}")
    return text


def gateway_chat(
    prompt: str,
    *,
    max_tokens: int = 4096,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    log: Callable[[str], None] | None = None,
    deployment: str | None = None,
) -> str:
    """Chat through the selected Azure, Gemini, or direct OpenAI backend."""
    configured = gateway_backend()
    active = active_gateway_backend()
    if not gateway_available():
        credential = (
            "OPENAI_API_KEY"
            if active == "openai"
            else "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT"
        )
        raise RuntimeError(f"Gateway LLM unavailable — set {credential} in .env")
    model = (deployment or gateway_deployment()).strip()
    if active == "gemini":
        temp = 0.2 if temperature is None else temperature
        text = gemini_chat(prompt, model=model, max_tokens=max_tokens, temperature=temp).strip()
        if log and text:
            preview = text.replace("\n", " ")
            log(f"  [{model}] response preview: {preview[:400]}{'…' if len(preview) > 400 else ''}")
        return text

    if active == "openai":
        return _openai_chat(
            prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            log=log,
        )

    from VisualHistory.llm import chat_once

    large = max_tokens >= 8192
    re = (
        reasoning_effort
        if reasoning_effort is not None
        else gateway_reasoning_effort(large_completion=large)
    )
    buffer = REASONING_OUTPUT_BUFFER if re else 0
    try:
        return chat_once(
            prompt,
            deployment=model,
            max_completion_tokens=max_tokens + buffer,
            temperature=None,
            reasoning_effort=re,
            log=log,
        ).strip()
    except Exception as exc:
        if configured != "azure-openai-fallback":
            raise
        activate_openai_fallback()
        if log:
            log(f"  Azure request failed ({exc}); switching permanently to direct OpenAI")
        return _openai_chat(
            prompt,
            model=deployment or gateway_openai_model(),
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            log=log,
        )


def apply_gateway_env(*, backend: str, deployment: str | None = None) -> None:
    """Set env vars so all cloud pipeline roles follow one gateway backend."""
    chosen = _normalize_gateway_backend(backend)
    if chosen == "azure-openai-fallback":
        _OPENAI_FAILOVER_ACTIVE.clear()
    os.environ["LITERATURE_GATEWAY_BACKEND"] = chosen
    os.environ["LITERATURE_ORCHESTRATOR_BACKEND"] = chosen
    os.environ["LITERATURE_SYNTHESIS_BACKEND"] = chosen
    if deployment:
        dep = deployment.strip()
        os.environ["LITERATURE_GATEWAY_DEPLOYMENT"] = dep
        os.environ["LITERATURE_ORCHESTRATOR_DEPLOYMENT"] = dep
        os.environ["LITERATURE_ASK_ANSWER_DEPLOYMENT"] = dep
        if chosen == "gemini":
            os.environ["AZURE_GEMINI_DEPLOYMENT"] = dep
            os.environ["GEMINI_DEPLOYMENT"] = dep
        elif chosen == "openai":
            os.environ["LITERATURE_OPENAI_MODEL"] = dep
            os.environ["OPENAI_MODEL"] = dep
        elif chosen == "azure-openai-fallback":
            os.environ["LITERATURE_AZURE_DEPLOYMENT"] = dep
            os.environ["LITERATURE_OPENAI_MODEL"] = dep
            os.environ["OPENAI_MODEL"] = dep
        else:
            os.environ["LITERATURE_AZURE_DEPLOYMENT"] = dep
