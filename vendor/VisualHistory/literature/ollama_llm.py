"""Minimal Ollama stub for Colab (OpenAI gateway path does not need local Ollama)."""

from __future__ import annotations


class ChatModelNotFoundError(RuntimeError):
    """Raised when the configured Ollama chat model is not installed."""


def default_chat_model() -> str:
    return "qwen2.5:7b"


def chat_model_available(model: str | None = None) -> bool:
    return False


def ollama_chat(*args, **kwargs) -> str:
    raise ChatModelNotFoundError("Ollama is not available in this portable Colab environment")
