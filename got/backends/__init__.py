"""LLM backends. Heavy deps are imported lazily so this stays cheap."""

from .base import AbstractLanguageModel, UsageStats
from .mock import MockLM


def get_backend(name: str, **kwargs) -> AbstractLanguageModel:
    """Factory: build a backend by name.

    Keeps experiment scripts free of import gymnastics -- a config file can
    just say ``backend: mock`` or ``backend: vllm``.
    """
    name = name.lower()
    if name == "mock":
        return MockLM(**kwargs)
    if name in ("llamacpp", "llama_cpp", "gguf"):
        from .local_models import LlamaCppLM
        return LlamaCppLM(**kwargs)
    if name in ("hf", "huggingface", "transformers"):
        from .local_models import HFLM
        return HFLM(**kwargs)
    if name == "vllm":
        from .local_models import VLLMLM
        return VLLMLM(**kwargs)
    raise ValueError(
        f"Unknown backend {name!r}. Choose from: mock, llamacpp, hf, vllm"
    )


__all__ = ["AbstractLanguageModel", "UsageStats", "MockLM", "get_backend"]
