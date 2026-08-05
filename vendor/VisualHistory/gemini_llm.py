"""Gemini via HKU Vertex AI gateway (see vertex-ai-gemini-service-api.yaml)."""

from __future__ import annotations

import os
from typing import Any

import requests

DEFAULT_GEMINI_LABEL = "Gemini 3.1 Pro"
DEFAULT_GEMINI_DEPLOYMENT = "gemini-3.1-pro-preview"
DEFAULT_VERTEX_GEMINI_BASE = "https://api.hku.hk/vertexai-gemini"

# Text/chat models (Gemini chat API in HKU portal).
TEXT_GEMINI_MODELS = frozenset(
    {
        "gemini-2.5-flash",
        "gemini-3-flash-preview",
        "gemini-3.1-pro-preview",
        "gemini-3.5-flash",
    }
)

# Image generation models — not for literature text pipeline.
IMAGE_GEMINI_MODELS = frozenset(
    {
        "gemini-2.5-flash-image",
        "gemini-3.1-flash-image",
        "gemini-3-pro-image",
    }
)

_KNOWN_GEMINI_SLUGS = TEXT_GEMINI_MODELS | IMAGE_GEMINI_MODELS


class GeminiNotConfiguredError(RuntimeError):
    """Raised when HKU gateway credentials are missing for Gemini."""


def _api_key() -> str:
    from VisualHistory.llm import API_KEY

    return API_KEY.strip()


def vertex_gemini_base_url() -> str:
    return (
        os.getenv("VERTEX_GEMINI_BASE_URL")
        or os.getenv("HKU_VERTEX_GEMINI_URL")
        or DEFAULT_VERTEX_GEMINI_BASE
    ).strip().rstrip("/")


def gemini_available() -> bool:
    return bool(_api_key())


def default_gemini_model() -> str:
    """HKU Vertex deployment id for Gemini text completions."""
    return (
        os.getenv("AZURE_GEMINI_DEPLOYMENT")
        or os.getenv("GEMINI_MODEL")
        or os.getenv("GEMINI_DEPLOYMENT")
        or DEFAULT_GEMINI_DEPLOYMENT
    ).strip()


def gemini_model_label(deployment: str | None = None) -> str:
    """Human-readable Gemini model name for notebook output and traces."""
    custom = os.getenv("GEMINI_MODEL_LABEL") or os.getenv("GEMINI_DISPLAY_NAME")
    if custom:
        return custom.strip()
    dep = (deployment or default_gemini_model()).strip()
    if dep in TEXT_GEMINI_MODELS or dep == DEFAULT_GEMINI_LABEL:
        if dep.startswith("gemini-3.1-pro"):
            return DEFAULT_GEMINI_LABEL
        return dep
    if dep.lower() in _KNOWN_GEMINI_SLUGS:
        return DEFAULT_GEMINI_LABEL
    return dep or DEFAULT_GEMINI_LABEL


def default_gemini_agent_model() -> str:
    return (
        os.getenv("AZURE_GEMINI_AGENT_DEPLOYMENT")
        or os.getenv("GEMINI_AGENT_MODEL")
        or default_gemini_model()
    ).strip()


def _extract_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        return ""
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    texts: list[str] = []
    for part in parts:
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    return "\n".join(texts).strip()


def gemini_chat(
    prompt: str,
    *,
    model: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.2,
) -> str:
    """Single-turn Gemini completion via HKU Vertex AI ``:generateContent``."""
    if not gemini_available():
        raise GeminiNotConfiguredError(
            "Gemini unavailable — set AZURE_OPENAI_API_KEY (HKU api-key header) in .env"
        )

    deployment = (model or default_gemini_model()).strip()
    if deployment in IMAGE_GEMINI_MODELS:
        raise ValueError(
            f"{deployment!r} is an image-generation model (Nano Banana). "
            f"Use a text model such as {DEFAULT_GEMINI_DEPLOYMENT!r} for literature/chat."
        )

    # Thinking models need headroom beyond internal reasoning tokens.
    output_tokens = max(max_tokens, 256)

    url = f"{vertex_gemini_base_url()}/{deployment}:generateContent"
    body: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": output_tokens,
        },
    }
    thinking_budget = os.getenv("GEMINI_THINKING_BUDGET", "").strip()
    if thinking_budget.isdigit():
        body["generationConfig"]["thinkingConfig"] = {
            "thinkingBudget": int(thinking_budget),
        }

    try:
        resp = requests.post(
            url,
            json=body,
            headers={"api-key": _api_key(), "Content-Type": "application/json"},
            timeout=float(os.getenv("GEMINI_REQUEST_TIMEOUT", "120")),
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"HKU Vertex Gemini request failed ({deployment}): {exc}") from exc

    if resp.status_code == 429 or resp.status_code == 403:
        detail = resp.text[:400]
        raise RuntimeError(
            f"HKU Vertex Gemini quota/rate limit ({deployment}, HTTP {resp.status_code}). {detail}"
        )
    if resp.status_code >= 400:
        raise RuntimeError(
            f"HKU Vertex Gemini error ({deployment}, HTTP {resp.status_code}): {resp.text[:400]}"
        )

    payload = resp.json()
    text = _extract_text(payload)
    if not text:
        raise RuntimeError(
            f"HKU Vertex Gemini returned empty text ({deployment}). "
            f"Response keys: {list(payload.keys())}"
        )
    return text
