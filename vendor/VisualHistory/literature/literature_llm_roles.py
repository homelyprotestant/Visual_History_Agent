"""Role-based LLM routing for the topic literature pipeline.

Cloud roles (orchestrator, ask, answer, cloud RCS, Finalize) share one gateway
backend: ``azure``, ``gemini``, or direct ``openai`` — set ``LITERATURE_GATEWAY_BACKEND``.
RCS may stay on local Ollama via ``LITERATURE_RCS_BACKEND=ollama``.
"""

from __future__ import annotations

import os
from typing import Callable, Literal

from VisualHistory.literature.gateway_llm import (
    gateway_available,
    gateway_backend,
    gateway_chat,
    gateway_deployment,
    gateway_label,
    gateway_reasoning_effort,
)
from VisualHistory.literature.ollama_llm import (
    ChatModelNotFoundError,
    chat_model_available,
    default_chat_model,
    ollama_chat,
)

Role = Literal["orchestrator", "rcs", "ask", "answer"]
RCSBackend = Literal["ollama", "azure", "gemini", "openai", "azure-openai-fallback"]


def orchestrator_deployment() -> str:
    return gateway_deployment()


def orchestrator_backend() -> Literal[
    "azure", "gemini", "openai", "azure-openai-fallback"
]:
    return gateway_backend()


def orchestrator_model() -> str:
    return gateway_deployment()


def orchestrator_label() -> str:
    return gateway_label()


def ask_answer_deployment() -> str:
    return gateway_deployment()


def rcs_backend() -> RCSBackend:
    explicit = (os.getenv("LITERATURE_RCS_BACKEND") or "").strip().lower()
    if explicit in {"ollama", "local", "qwen"}:
        return "ollama"
    if explicit in {"gemini", "google"}:
        return "gemini"
    if explicit in {"azure", "openai", "gpt", "cloud", "gateway"}:
        return gateway_backend()
    # Default: local RCS unless LITERATURE_RCS_LOCAL=0
    if os.getenv("LITERATURE_RCS_LOCAL", "1").strip().lower() in {"0", "false", "no"}:
        return gateway_backend()
    return "ollama"


def rcs_model() -> str:
    backend = rcs_backend()
    if backend in {"azure", "gemini", "openai", "azure-openai-fallback"}:
        return (
            os.getenv("LITERATURE_RCS_DEPLOYMENT")
            or os.getenv("LITERATURE_RCS_MODEL")
            or gateway_deployment()
        ).strip()
    return (
        os.getenv("LITERATURE_RCS_MODEL")
        or os.getenv("OLLAMA_MODEL")
        or os.getenv("LOCAL_LLM_MODEL")
        or default_chat_model()
    ).strip()


def rcs_reasoning_effort() -> str:
    return gateway_reasoning_effort()


def role_available(role: Role) -> bool:
    if role in {"orchestrator", "ask", "answer"}:
        return gateway_available()
    if rcs_backend() in {"azure", "gemini", "openai", "azure-openai-fallback"}:
        return gateway_available()
    return chat_model_available(rcs_model())


def role_chat(
    role: Role,
    prompt: str,
    *,
    max_tokens: int = 4096,
    temperature: float | None = None,
    log: Callable[[str], None] | None = None,
) -> str:
    """Dispatch a prompt to the model assigned to ``role``."""
    if role in {"orchestrator", "ask", "answer"}:
        if not gateway_available():
            raise ChatModelNotFoundError(
                f"Configured {gateway_backend()} gateway unavailable for {role}"
            )
        temp = 0.2 if role == "orchestrator" else None
        if temperature is not None:
            temp = temperature
        return gateway_chat(
            prompt,
            max_tokens=max_tokens,
            temperature=temp,
            log=log,
        )

    model = rcs_model()
    if rcs_backend() in {"azure", "gemini", "openai", "azure-openai-fallback"}:
        if not gateway_available():
            raise ChatModelNotFoundError(
                f"Configured {gateway_backend()} gateway unavailable for RCS"
            )
        temp = 0.1 if temperature is None else temperature
        return gateway_chat(
            prompt,
            max_tokens=max_tokens,
            temperature=temp,
            reasoning_effort=(
                gateway_reasoning_effort()
                if rcs_backend() in {"azure", "openai", "azure-openai-fallback"}
                else None
            ),
            deployment=model,
        )

    if not chat_model_available(model):
        raise ChatModelNotFoundError(f"RCS model {model!r} unavailable — run: ollama pull {model}")
    temp = 0.1 if temperature is None else temperature
    return ollama_chat(prompt, model=model, max_tokens=max_tokens, temperature=temp).strip()
